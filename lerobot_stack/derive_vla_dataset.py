"""Derive the pi0.5 dataset from an ACT corpus (MS7h design 7.1).

LeRobot 0.5.1's ``pi05`` reads the images and ``observation.state`` only, so the 16-wide context
moves into the state: the derived ``observation.state`` is the 7 state values and then the 16
``observation.environment_state`` values, the camera roll wrapped as the ACT context fix serves it
(``policy_server.wrap_context_roll``, MS7g D12), and ``observation.environment_state`` leaves the
dataset. Every episode gets the one task string ``TASK``: pi05 writes it into its prompt.

The statistics of the state and the action are computed over every kept frame, quantiles too
(LeRobot's ``get_feature_stats`` on the whole column): pi05 maps both to [-1, 1] by q01 and q99,
and the corpus's own quantiles are the frame-weighted mean of the per-episode quantiles
(``aggregate_stats``). For a per-episode constant such as the camera pose that collapses q01 and
q99 onto the mean, and for the action it is about a third of the true spread (2026-10-05, the
whole corpus: joint 1 q01 -0.601 against -1.779, so the normalized joint actions reached +-4).
The per-episode state stats in meta/episodes are recomputed too; every other feature's stats are
copied. The videos are hard links to the source files (one filesystem
needed), the images and the action are untouched, and the source is never written.

``--episode_fraction P/Q`` keeps the first ceil(n P/Q) episodes of each wave of n (MS7h design 5.4
step 2: the Drive copy when space is short). The waves come from the merge ledgers that
merge_roots.sh writes next to every merged root: a root named ``dataset`` with a
``<root>.ledger.json`` expands into its roots in order, and any other root ``.../w<N>/<name>`` is
wave ``w<N>``. A contiguous block, not every third episode, because one video file holds about 100
consecutive episodes, so a spread subset would still need every file. The subset is renumbered;
the stats of the index columns (``index``, ``episode_index``) stay the source's.

Run in the lerobot venv:
    lerobot_stack/.venv/bin/python lerobot_stack/derive_vla_dataset.py \\
        --src ~/data/maker-arm-ms7/g/w1234567/dataset --out ~/data/maker-arm-ms7/h/vla/dataset
-> one DERIVE_OK / DERIVE_FAIL line.
"""

import argparse
import json
import math
import os
import shutil
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from policy_server import CONTEXT_ROLL, wrap_context_roll

STATE = "observation.state"
ENV = "observation.environment_state"
TASK = "pick up the marked part and place it in its box"
ROLL_NAME = "overhead_cam_roll"


def read_tables(directory: Path, names: list[str] | None = None) -> pa.Table:
    files = sorted(Path(directory).glob("*/*.parquet"))
    return pa.concat_tables([pq.read_table(f, columns=names) for f in files])


def waves(root: Path) -> list[tuple[str, np.ndarray]]:
    """[(wave, episode lengths)] of ``root`` in its merge order, adjacent runs of a wave joined."""
    root = Path(root)
    ledger = Path(f"{root}.ledger.json")
    if root.name != "dataset" or not ledger.is_file():
        lengths = read_tables(root / "meta/episodes", ["length"])["length"].to_numpy()
        return [(root.parent.name, lengths)]
    out: list[tuple[str, np.ndarray]] = []
    for sub in json.loads(ledger.read_text())["roots"]:
        for wave, lengths in waves(Path(sub)):
            if out and out[-1][0] == wave:
                out[-1] = (wave, np.concatenate([out[-1][1], lengths]))
            else:
                out.append((wave, lengths))
    return out


def keep_mask(sizes: list[int], fraction: Fraction) -> np.ndarray:
    """One bool an episode: the first ceil(n * fraction) episodes of each wave of n."""
    return np.concatenate([np.arange(n) < math.ceil(n * fraction) for n in sizes])


def columns(table: pa.Table, name: str) -> np.ndarray:
    """A list<float> column as an (n, width) float32 array."""
    column = table[name].combine_chunks()
    return np.asarray(column.flatten(), dtype=np.float32).reshape(len(column), -1)


def derived_state(state: np.ndarray, context: np.ndarray) -> np.ndarray:
    """(n, 7) state + (n, 16) raw context -> (n, 23), the roll wrapped as the ACT server serves it."""
    return np.concatenate([state, wrap_context_roll(context)], axis=1).astype(np.float32)


def feature_stats(array: np.ndarray) -> dict:
    from lerobot.datasets.compute_stats import get_feature_stats

    return get_feature_stats(array.astype(np.float64), axis=0, keepdims=False)


def put(table: pa.Table, name: str, values) -> pa.Table:
    """``table`` with column ``name`` replaced by ``values``, or appended."""
    i = table.schema.get_field_index(name)
    array = pa.array(values)
    return table.set_column(i, name, array) if i >= 0 else table.append_column(name, array)


def derive(src: Path, out: Path, fraction: Fraction = Fraction(1)) -> dict:
    src, out = Path(src), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    info = json.loads((src / "meta/info.json").read_text())
    names = info["features"][ENV]["names"]
    if names[CONTEXT_ROLL] != ROLL_NAME:
        raise ValueError(f"{ENV}[{CONTEXT_ROLL}] is {names[CONTEXT_ROLL]}, not {ROLL_NAME}")
    episodes = read_tables(src / "meta/episodes")
    plan = waves(src)
    sizes = [len(lengths) for _, lengths in plan]
    if not np.array_equal(np.concatenate([x for _, x in plan]), episodes["length"].to_numpy()):
        raise ValueError(f"the merge ledgers of {src} do not add up to its episodes")
    keep = keep_mask(sizes, fraction)
    renumber = np.cumsum(keep) - 1  # source episode -> derived episode, where kept

    out.mkdir(parents=True)
    states, actions, offset = [], [], 0
    for path in sorted((src / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        index = table["index"].to_numpy()
        if not np.array_equal(index, np.arange(offset, offset + len(index))):
            raise ValueError(f"{path} does not continue the frame index at {offset}")
        offset += len(index)
        table = table.filter(pa.array(keep[table["episode_index"].to_numpy()]))
        if not table.num_rows:
            continue
        state = derived_state(columns(table, STATE), columns(table, ENV))
        n, width = state.shape
        done = sum(len(s) for s in states)
        states.append(state)
        actions.append(columns(table, "action"))
        offsets = pa.array(np.arange(0, (n + 1) * width, width, dtype=np.int32))
        flat = pa.array(state.ravel())
        table = put(table, STATE, pa.ListArray.from_arrays(offsets, flat, table[STATE].type))
        table = put(table, "episode_index", renumber[table["episode_index"].to_numpy()])
        table = put(table, "index", np.arange(done, done + n)).drop_columns([ENV])
        target = out / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), target)
    state = np.concatenate(states)

    kept = episodes.filter(pa.array(keep))
    lengths = kept["length"].to_numpy()
    ends = np.cumsum(lengths)
    if ends[-1] != len(state):
        raise ValueError(f"{len(state)} frames written, the kept episodes hold {ends[-1]}")
    per_episode = [feature_stats(state[b - n : b]) for n, b in zip(lengths, ends)]
    stale = [c for c in kept.column_names if c.startswith((f"stats/{STATE}/", f"stats/{ENV}/"))]
    kept = kept.drop_columns(stale)
    for key in per_episode[0]:
        kept = put(kept, f"stats/{STATE}/{key}", [s[key].tolist() for s in per_episode])
    m = kept.num_rows
    kept = put(kept, "episode_index", np.arange(m))
    kept = put(kept, "tasks", [[TASK]] * m)
    kept = put(kept, "dataset_from_index", ends - lengths)
    kept = put(kept, "dataset_to_index", ends)
    kept = put(kept, "meta/episodes/chunk_index", np.zeros(m, np.int64))
    kept = put(kept, "meta/episodes/file_index", np.zeros(m, np.int64))
    (out / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        kept.replace_schema_metadata(None), out / "meta/episodes/chunk-000/file-000.parquet"
    )
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(
        out / "meta/tasks.parquet"
    )

    stats = json.loads((src / "meta/stats.json").read_text())
    stats.pop(ENV)
    for key, values in ((STATE, state), ("action", np.concatenate(actions))):
        stats[key] = {k: v.tolist() for k, v in feature_stats(values).items()}
    (out / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    wrapped = [f"{ROLL_NAME}_wrapped" if n == ROLL_NAME else n for n in names]
    info["features"].pop(ENV)
    info["features"][STATE] = {
        "dtype": "float32",
        "shape": [state.shape[1]],
        "names": info["features"][STATE]["names"] + wrapped,
    }
    info.update(total_episodes=m, total_frames=len(state), total_tasks=1)
    info["splits"] = {"train": f"0:{m}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    if (src / "meta/action_space.json").is_file():
        shutil.copy(src / "meta/action_space.json", out / "meta/action_space.json")

    videos = 0
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        chunks, files = (
            kept[f"videos/{key}/{c}"].to_numpy() for c in ("chunk_index", "file_index")
        )
        for chunk, file in sorted(set(zip(chunks, files))):
            rel = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(src / rel, out / rel)
            videos += 1
    split = np.split(keep, np.cumsum(sizes)[:-1])
    kept_waves = ",".join(f"{w}:{int(k.sum())}/{len(k)}" for (w, _), k in zip(plan, split))
    return {"episodes": m, "frames": len(state), "videos": videos, "waves": kept_waves}


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"DERIVE_FAIL usage: {message}", flush=True)
        os._exit(2)


def main() -> int:
    parser = TokenArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, type=Path, help="the ACT corpus root (read only)")
    parser.add_argument("--out", required=True, type=Path, help="the new root; must not exist")
    parser.add_argument("--episode_fraction", type=Fraction, default=Fraction(1))
    try:
        args = parser.parse_args()
        if not 0 < args.episode_fraction <= 1:
            raise ValueError(f"--episode_fraction must be in (0, 1], got {args.episode_fraction}")
        result = derive(args.src.expanduser(), args.out.expanduser(), args.episode_fraction)
    except Exception as exc:  # noqa: BLE001 -- any failure must become DERIVE_FAIL
        traceback.print_exc()
        print(f"DERIVE_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    summary = " ".join(f"{k}={v}" for k, v in result.items())
    print(f"DERIVE_OK {summary} fraction={args.episode_fraction} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())

# MA0 — Bench Prep (gate report)

Date: 2026-08-20

## Gate checklist (registered in the system design, 2026-08-20)

| Item | Result | Evidence |
|------|--------|----------|
| Rust protocol crate passes ported official-SDK vectors | PASS | `cargo test -p maker-arm-protocol` on Thor (HEAD `101616d`): 16 passed; 0 failed; 0 ignored. Includes `round_half_to_even_is_load_bearing`, `rs02_ranges_and_firmware_trap`, and `non_finite_input_is_pinned_not_rejected`. |
| vcan integration green in CI | PASS | CI run: https://github.com/youtalk/maker-arm-rs/actions/runs/32433778128 (`vcan` job green, ran `cargo test -p maker-arm-transport --features socketcan -- --ignored`). Re-run locally on Thor against real `vcan0` today: 6 passed, 0 failed. |
| aarch64 wheel builds + golden-parity pytest | PASS | CI run: https://github.com/youtalk/maker-arm-rs/actions/runs/32433778128 (`wheels (ubuntu-24.04-arm)` job green — builds the wheel via `maturin-action` and runs `pytest bindings/maker-arm-py/tests/`). Thor-native (Task 11 Step 5): built `maker_arm_rs-0.1.0-cp312-cp312-linux_aarch64.whl` via `maturin develop`, `pytest` — 6 passed. Module resolves to `.venv/.../site-packages/maker_arm_rs/__init__.py` (correct name `maker_arm_rs`, not `maker_arm`). |
| Thor PREEMPT_RT kernel active | **FAIL** | `uname -a`: `Linux thor 6.8.12-tegra #1 SMP PREEMPT Tue Dec 30 15:40:41 PST 2025 aarch64 aarch64 aarch64 GNU/Linux`. `/sys/kernel/realtime`: does not exist. See "Notes and deviations" for the full explanation — this is the stock PREEMPT (voluntary) kernel, not PREEMPT_RT. |
| cyclictest baseline recorded | **NOT RUN** | `cyclictest` requires root; the operating user's `ulimit -r` is `0` and `sudo -n true` fails ("a password is required"), so it cannot run unprivileged in this session. The controller has asked the owner to run it manually; the result has not come back at the time of writing. Pending command: `sudo cyclictest -m -Sp90 -i200 -l100000 -h400 -q > ~/cyclictest-baseline.txt` |
| SocketCAN sanity (vcan roundtrip on Thor) | PASS | Task 11 Step 4, `cansend`/`candump` roundtrip on `vcan0`: `vcan0  01800001   [8]  80 00 80 00 05 1F 33 33` — byte-identical to the golden MIT vector from the protocol tests. Reconfirmed today: `cargo test -p maker-arm-transport --features socketcan -- --ignored` against live `vcan0` (interface state `<NOARP,UP,LOWER_UP> mtu 72`) — 6 passed, including `roundtrip_extended_frame_over_vcan`, `remote_frames_are_not_delivered_as_motor_data`, and `standard_id_frames_are_not_delivered_as_motor_data`. |
| maker-arm-lab CI green | PASS | CI run: https://github.com/youtalk/maker-arm-lab/actions/runs/32431942689 — `test` job green (post-fix commit `e7751f1`, which made the package installable and derived `__version__` from package metadata). |

## cyclictest detail

Not available. The baseline run is pending — see the "NOT RUN" row above for
the exact reason (requires root; `ulimit -r` is 0 for the operating user) and
the exact pending command. The histogram summary (min/avg/max per priority
class) will be pasted here once `~/cyclictest-baseline.txt` exists.

**This gate item, when it lands, will be a stock-PREEMPT baseline, not the
PREEMPT_RT baseline the plan originally intended** (see the "Thor
PREEMPT_RT kernel active" FAIL above). This matters for any later milestone
that compares a tuned kernel/scheduler configuration against "the MA0
baseline" — that comparison will be stock-PREEMPT vs. tuned, not
PREEMPT_RT-vs-tuned, and should be labeled as such wherever it is used.

## Notes and deviations

### Thor PREEMPT_RT kernel — why this is a FAIL, not a skip

The plan's Task 11 Step 1 specified installing the RT kernel via:

```bash
sudo apt install -y nvidia-l4t-rt-kernel nvidia-l4t-rt-kernel-headers \
  nvidia-l4t-rt-kernel-oot-modules nvidia-l4t-display-rt-kernel \
  nvidia-l4t-rt-kernel-openrm
sudo reboot
```

None of those five packages exist in the JetPack 7 / L4T r38.4 apt index.
`apt-cache search nvidia-l4t-rt` on this host returns nothing. The r38.4
index (`https://repo.download.nvidia.com/jetson/som r38.4/main`) ships only
the `nvidia-l4t-kernel*` family (`nvidia-l4t-kernel`,
`nvidia-l4t-kernel-dtbs`, `nvidia-l4t-kernel-headers`,
`nvidia-l4t-kernel-oot-headers`, `nvidia-l4t-kernel-oot-modules`,
`nvidia-l4t-kernel-openrm`, …), all currently installed at version
`6.8.12-tegra-38.4.0-20251230160601`. The `nvidia-l4t-rt-kernel*` package set
the plan names is the JetPack 6 / L4T r36 path and does not carry forward to
r38.4. This was verified against the r38.4 apt index fetched 2026-08-05
(controller pre-flight) and reconfirmed directly on the host while writing
this report.

Running kernel identity:

```
Linux thor 6.8.12-tegra #1 SMP PREEMPT Tue Dec 30 15:40:41 PST 2025 aarch64 aarch64 aarch64 GNU/Linux
```

`/sys/kernel/realtime` is absent, confirming this is the stock PREEMPT
(voluntary preemption) kernel, not PREEMPT_RT.

**Owner decision**: skip the RT kernel install, run the cyclictest baseline
on the stock kernel instead, execute the rest of the Task 11 hardware
checklist (Steps 2–5) in full, and record this gate row as FAIL with the
evidence above — not softened, not deleted. No reboot was performed.

### Other deviations from the plan

- GitHub repositories were created **private**, not public — the plan
  specified `--public`; this was an explicit owner decision that overrides
  the plan.
- CI defect in the plan's own YAML: the `vcan` job's `modprobe vcan` failed
  on GitHub's hosted Azure runner kernel
  (`linux-modules-extra` ships the `vcan` module separately from the base
  image on `6.17.0-1022-azure`). Fixed in commit `3288067`
  (`fix(ci): install linux-modules-extra so vcan loads on GitHub runners`).
  CI green thereafter: run 32425539801.
- No pyo3 version bump was needed (pinned at 0.24, verified compatible with
  the local Python 3.12.3); the anticipated `extension-module`/`cargo test`
  link conflict did not occur in practice (see "Carried forward to MA1"
  below for why it's still a latent risk).
- `sudo` on the Thor host requires an interactive password
  (`sudo -n` fails); `rt-tests` and `can-utils` installation and the
  `vcan0` interface bring-up (`modprobe vcan`, `ip link add`) were run
  interactively by the owner, once, ahead of the automated task work.
- Toolchain actually used: `rustc`/`cargo` 1.98.0 (aarch64), `uv` 0.12.5.

## Carried forward to MA1

### Open design questions from this milestone (both now addressed, not resolved)

- **`ReplayBackend::from_log` silently dropped unparseable lines.** For a
  golden-trace replay tool whose whole purpose is candump fidelity, a
  malformed line most plausibly means a corrupt capture, and silently
  dropping it yields a shorter, silently-different frame sequence — a
  parity test could then pass or fail for the wrong reason with no
  diagnostic. **Mitigated, not removed**, in the final fix wave (commit
  `4610e3a`): `from_log` still parses forgivingly but now records every
  dropped line via `SkippedLine { number, text }`, retrievable through
  `skipped()`; a new `from_log_strict()` returns a `Result` instead and a
  `remaining()` accessor was added. Logs that parsed cleanly before still
  parse the same way. MA1 (or the parity tooling that consumes replay logs)
  should decide whether golden-trace parity checks should use
  `from_log_strict` rather than the forgiving path.
- **Standard/extended CAN id conflation in `SocketCanBackend::recv`.**
  `recv` originally folded standard 11-bit and extended 29-bit ids into the
  same `u32` space (standard `0x001` and extended `0x00000001` both
  surfaced as `(1, data)`, indistinguishable), and installed no `CAN_RAW`
  filter, so stray standard or RTR traffic could be reinterpreted as valid
  motor data. **Fixed** in the final fix wave (commit `53d6592`):
  `accept_frame` now takes extended-id data frames only, enforced both in
  userspace and by a kernel filter (`CAN_EFF_FLAG`,
  `CAN_EFF_FLAG | CAN_RTR_FLAG`) installed at `open()`. Regression tests:
  `standard_id_frames_are_not_delivered_as_motor_data`,
  `remote_frames_are_not_delivered_as_motor_data`,
  `rejects_standard_id_frames`, `rejects_remote_frames`,
  `rejects_standard_id_remote_frames`.

### Items explicitly deferred to MA1 by controller decision

- **Carrying DLC through the `CanBackend` trait.** `Frame.data` is
  currently always 8 bytes (every private-protocol frame is DLC 8); DLC is
  a cross-cutting API change to the trait shared by mock, replay, and
  SocketCAN backends, deferred rather than done piecemeal.
- **`READ_VERSION` / `GET_ID` encoders.** Not implemented in MA0. These are
  the frames the RS02 firmware check (below) and general bring-up
  diagnostics depend on.
- **`TransportError::Io` should hold a real `std::io::Error`, not a
  `String`.** The current `Io(String)` variant discards the underlying
  `ErrorKind`. This was workable in MA0 only because the SocketCAN timeout
  path reads `e.kind()` *before* the string conversion — any future
  ErrorKind-based branching downstream of `TransportError` needs the same
  care, or needs this fixed first.
- **Gate pyo3's `extension-module` feature behind a cargo feature**, rather
  than enabling it unconditionally. It works today only because
  `bindings/maker-arm-py/src/lib.rs` has zero `#[test]` items, so the
  PyO3-touching code is unreachable from the synthetic test harness and
  `--gc-sections` discards it before symbol resolution. Adding a `#[test]`
  to that crate would resurface the `cargo test --all-features` link
  failure this was masking.

### Nothing composes the two crates end-to-end

No test in this milestone exercises the full path encode → send → recv →
parse across both crates. `maker-arm-transport` is deliberately forbidden
from depending on `maker-arm-protocol` (layering rule), so there is no
crate in MA0 where an encode-then-decode round trip over a real or mock
backend can legitimately live. The natural home for that test is MA1's
session crate, which depends on both.

### `float_to_u16` non-finite input — documented and tested, not resolved

`float_to_u16` cannot fail: non-finite input silently lands on a range
extreme rather than erroring.

- `NaN` clamps to the range **minimum** (e.g. for position, `P_MIN` =
  −12.57 rad).
- `+inf` clamps to the range **maximum**; `-inf` to the minimum.

This diverges from the official Python SDK, whose `int(round(x))` raises
`ValueError` on NaN and `OverflowError` on ±inf. The behavior is now pinned
by the Rust test `non_finite_input_is_pinned_not_rejected` and documented on
`float_to_u16`'s doc comment, but it is **not resolved**: MA1 must decide
whether to validate, error, or clamp at the point commanded values enter the
system, before any IK or PD-loop math feeds numbers into this encoder — a
NaN from an upstream computation would otherwise silently become a
full-negative-rail position command rather than a visible failure.

### `SocketCanBackend` loopback

SocketCAN delivers a socket's own sent frames back to *other* sockets bound
to the same interface (CAN_RAW loopback is a kernel-level, per-interface
behavior, not per-socket). An MA1 session that opens separate tx and rx
sockets on the same interface would receive its own outgoing commands on
the rx socket. In particular, `parse_frame` would decode an echoed Type-17
(param read request) frame as a `ParamReply` — a self-inflicted "reply"
that never came from the motor. MA1's session layer needs either a single
shared socket, an id/source discriminator, or an explicit loopback-disable
setsockopt.

### RS02 firmware check and golden trace capture (already named in the brief)

- **RS02 firmware version check (`READ_VERSION`) before anything moves.**
  `rs02_ranges_and_firmware_trap` pins the assumption that the RS02
  position range is ±12.57 rad on firmware > 0.2.2.11 (older firmware maps
  ±12.5 rad). This is currently unverified against real hardware and must
  be confirmed via `READ_VERSION` (itself deferred to MA1, see above)
  before trusting any position decode.
- **Golden CAN trace capture from the official SDK bring-up session** — not
  yet captured; needed to validate `ReplayBackend`/parity tooling against
  a real bring-up rather than only the ported unit vectors.

### Temperature decoded as unsigned — confirm during bring-up

`MotorFeedback::temperature` is decoded as an **unsigned** `u16` (raw byte
pair `/ 10.0`), matching the official SDK byte-for-byte, which is the hard
requirement this project was built against. The cost: a sub-zero reading
wraps instead of going negative — e.g. raw `0xFFF6` (two's-complement −1°C)
decodes as `6552.6°C`, a nonsense value that at least fails obviously rather
than looking plausible. MA1 must confirm the actual sign convention against
upstream (and real hardware) during bring-up — if the field is genuinely
`int16`, this needs to change before temperature readings are trusted for
anything beyond an over-temperature trip on an implausible value.

## Defects this milestone found in the plan itself

These are process findings worth keeping — each was a case where following
the plan's own text literally would have produced behavior contradicting
the plan's own stated intent, so each was treated as a defect to fix, not a
scope change to escalate.

- **CI `vcan` job failure.** GitHub's hosted Azure runner kernel
  (`6.17.0-1022-azure`) ships the `vcan` kernel module in the
  `linux-modules-extra` package, not the base image, so the plan's
  `modprobe vcan` step failed outright (CI run 32425375749). Fixed by
  adding `apt-get install linux-modules-extra-$(uname -r)` before the
  `modprobe` (commit `3288067`).
- **`parse_candump_line` panicked on multi-byte UTF-8 payloads.** The
  length guard checked only byte length
  (`data_hex.len() % 2 != 0 || data_hex.len() > 16`), then sliced at fixed
  2-byte offsets — a payload field containing a multi-byte UTF-8 character
  (e.g. "😀", 4 bytes, even length, ≤16) panics with "byte index 2 is not a
  char boundary". This parser consumes `candump -l` files captured off real
  hardware, i.e. untrusted input, and the function's own doc comment states
  it "Returns `None` on anything that isn't a well-formed frame line."
  Fixed in commit `1aeb5c0` by rejecting non-ASCII payload hex before
  slicing — proven to close the entire panic class structurally (ASCII byte
  offsets and char boundaries always coincide), not just the one repro.
- **`SocketCanBackend::recv` blocked forever on a zero timeout.** Linux
  interprets a zero `SO_RCVTIMEO` timeval as "no timeout", so
  `recv(Duration::ZERO)` blocked indefinitely instead of polling and
  returning immediately — demonstrated directly against a real socket
  (watchdog thread had to force-exit after the call was still blocked 1.5s
  later). Mock and replay backends return `Ok(None)` instantly for the same
  call, so any control loop written against the trait's apparent contract
  would hang on a quiet real bus. The same review also found `recv`
  accepted RTR and standard-id frames: an RTR frame's empty payload
  zero-pads to `[0u8; 8]`, and `parse_frame` turned one such frame into a
  structurally valid `MotorFeedback` reading every channel (position,
  velocity, torque) at its negative rail — exactly the kind of input a PD
  loop would convert into a large corrective command. Both fixed in commit
  `53d6592`: zero-timeout `recv` now polls via `poll(2)` instead of relying
  on `SO_RCVTIMEO` (measured directly: a 1µs `SO_RCVTIMEO` actually blocks
  ~8–10ms on this kernel due to jiffy rounding, unusable at 1kHz), and
  `accept_frame` now rejects non-extended and RTR frames both in userspace
  and via a kernel-installed `CAN_RAW` filter.
- **Round-half-to-even had no Rust test pinning it, despite being the
  project's single hardest numeric requirement.** The final whole-branch
  review mutated `round_ties_even` to `round()` (ordinary round-half-away-
  from-zero) and all 13 vectors then in the suite still passed — every tie
  the suite exercised at that point was exactly `32767.5`, where both
  rounding modes agree. Fixed by adding a canary vector that is not such a
  tie: `float_to_u16(14.0, RS02.t_min, RS02.t_max)` computes to
  `(14+17)*65535/34 = 59752.5` exactly, where half-to-even gives `59752`
  and half-away-from-zero gives `59753` — the same mutation against the
  fixed suite fails exactly on this one assertion, leaving the other 15
  green (commit `ee90a1f`).

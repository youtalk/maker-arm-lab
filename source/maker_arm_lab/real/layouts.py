"""What the iPad shows: its screen geometry and the wrist camera's checkerboard. Every image
is drawn one to one, so a 100 mm scale bar on it checks with a ruler."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Screen:
    """A screen of ``width_px`` x ``height_px`` at ``ppi``. Its frame has the origin at the
    screen centre, x along +u (the long side) and y along -v, in metres. Pixel coordinates use
    the lab convention: the centre of pixel (0, 0) is (0.5, 0.5)."""

    width_px: int
    height_px: int
    ppi: float

    @property
    def mm_per_px(self) -> float:
        return 25.4 / self.ppi

    def to_px(self, xy_m) -> tuple[float, float]:
        s = 1000.0 / self.mm_per_px
        return (self.width_px / 2 + xy_m[0] * s, self.height_px / 2 - xy_m[1] * s)

    def from_px(self, u: float, v: float) -> tuple[float, float]:
        s = self.mm_per_px / 1000.0
        return ((u - self.width_px / 2) * s, (self.height_px / 2 - v) * s)


IPAD_AIR_11 = Screen(2360, 1640, 264.0)


def checker_png(
    cols: int = 9, rows: int = 6, square_px: int = 208, screen: Screen = IPAD_AIR_11
) -> Image.Image:
    """A checkerboard of ``cols`` x ``rows`` inner corners (20.01 mm squares at 208 px on the
    iPad), centred on a white screen, with a 100 mm scale bar under it."""
    img = Image.new("RGB", (screen.width_px, screen.height_px), "white")
    draw = ImageDraw.Draw(img)
    w, h = (cols + 1) * square_px, (rows + 1) * square_px
    left, top = (screen.width_px - w) // 2, (screen.height_px - h) // 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x, y = left + c * square_px, top + r * square_px
                draw.rectangle([x, y, x + square_px - 1, y + square_px - 1], fill="black")
    bar = round(100.0 / screen.mm_per_px)
    x0, y = (screen.width_px - bar) // 2, (top + h + screen.height_px) // 2
    draw.rectangle([x0, y - 3, x0 + bar - 1, y + 2], fill="black")
    for x in (x0, x0 + bar - 1):
        draw.rectangle([x - 1, y - 12, x + 1, y + 11], fill="black")
    return img

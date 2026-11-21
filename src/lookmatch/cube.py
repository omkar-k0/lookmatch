"""Read and write .cube files (Resolve / Adobe 3D LUT format). Red changes fastest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lookmatch.types import Lut


def write(lut: Lut, path: str | Path, title: str = "lookmatch") -> None:
    n = lut.size
    lines = [f'TITLE "{title}"', f"# method {lut.method} version {lut.version}"]
    for k, v in sorted(lut.meta.items()):
        lines.append(f"# {k} {v}")
    lines += [f"LUT_3D_SIZE {n}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    t = np.clip(lut.table, 0.0, 1.0)
    body = t.transpose(2, 1, 0, 3).reshape(-1, 3)  # b slowest, r fastest
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in body]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(path: str | Path) -> Lut:
    n = None
    rows: list[list[float]] = []
    title = ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0]
        if key == "TITLE":
            title = line[5:].strip().strip('"')
        elif key == "LUT_3D_SIZE":
            n = int(line.split()[1])
        elif key in ("DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE", "LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
            if key == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported")
            continue
        else:
            rows.append([float(x) for x in line.split()[:3]])
    if n is None or len(rows) != n**3:
        raise ValueError(f"{path}: expected {n}^3 rows, got {len(rows)}")
    table = np.asarray(rows).reshape(n, n, n, 3).transpose(2, 1, 0, 3)
    return Lut(table, method="cube_file", meta={"title": title})

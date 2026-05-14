"""路径 / 文件名自然排序（frame2 < frame10）。"""

from __future__ import annotations

import re
from pathlib import Path


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

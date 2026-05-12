#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[INFO] quick 模式（约 2 分钟验证）"
python - <<'PY'
import yaml
from pathlib import Path
p = Path("config.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["train"]["mode"] = "quick"
p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("config.yaml 已切换为 quick")
PY

python src/run_pipeline.py --config config.yaml

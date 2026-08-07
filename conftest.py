"""Make the repo root importable so `pytest` finds `vnocr` without an install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

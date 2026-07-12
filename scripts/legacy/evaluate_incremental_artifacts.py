#!/usr/bin/env python3
"""Compatibility entry point for :mod:`scripts.evaluate_incremental_artifacts`.

The maintained implementation lives under ``scripts/``.  This wrapper keeps
older commands that invoke the repository-root path working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluate_incremental_artifacts import main


if __name__ == "__main__":
    main()

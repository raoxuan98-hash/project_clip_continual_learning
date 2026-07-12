#!/usr/bin/env python3
"""Compatibility entry point for the maintained RGDA sweep evaluator.

The implementation lives in ``scripts.evaluate_incremental_rgda_sweep_artifacts``.
This wrapper preserves older repository-root invocations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluate_incremental_rgda_sweep_artifacts import main


if __name__ == "__main__":
    main()

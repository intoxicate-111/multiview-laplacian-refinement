#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed Python/NumPy/PyTorch, then execute an unmodified Python entrypoint."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass
    sys.argv = [str(args.script), *remainder]
    runpy.run_path(str(args.script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Single entry point. Generate, validate, analyse. Reproducible from a clean checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import generate
import analyse

if __name__ == "__main__":
    print("=" * 68)
    print("1/2  generating simulated data")
    print("=" * 68)
    generate.main()
    print()
    print("=" * 68)
    print("2/2  cleaning, validating and analysing")
    print("=" * 68)
    analyse.main()

#!/usr/bin/env python
"""
Generate FCSTNorm2018.pkl normalization file required for forecasting.

This script calculates min, max, mean, and std statistics for each forecast field
and saves them to a pickle file that will be used during inference.

Usage:
    python generate_fcst_norm.py --year 2018
"""

import os
import sys
import argparse

# Set environment variable before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

sys.path.insert(1, "../")
from data.data_gefs import gen_fcst_norm

def main():
    parser = argparse.ArgumentParser(
        description='Generate forecast normalization constants'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2018,
        help='Year to use for calculating normalization statistics (default: 2018)'
    )

    args = parser.parse_args()

    print(f"Generating forecast normalization constants for year {args.year}...")
    print("This may take several minutes depending on the size of your data.")
    print()

    try:
        gen_fcst_norm(year=args.year)
        print()
        print("=" * 70)
        print(f"✓ Successfully generated FCSTNorm{args.year}.pkl")
        print("=" * 70)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Error generating normalization file: {str(e)}")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()

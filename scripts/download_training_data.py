#!/usr/bin/env python3
"""
Download training data from GCS bucket for cGAN model training.

This script downloads forecast, truth, and constant files from the sewaa-ifs-train
GCS bucket and organizes them according to the cGAN_tutorial config structure.

Usage:
    python download_training_data.py \\
        --creds /path/to/service-account.json \\
        --dest /path/to/local/data \\
        --years 2018 2019 2020 \\
        --skip-existing \\
        --workers 16

Example:
    python download_training_data.py \\
        --creds ~/service-account.json \\
        --dest /mnt/training-data \\
        --years 2018 2019 2020 2021 2023 \\
        --download-constants \\
        --skip-existing
"""

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
from typing import List, Tuple

from google.api_core.retry import Retry
from google.cloud import storage


# Default GCS bucket configuration
DEFAULT_BUCKET = "sewaa-ifs-train"
AVAILABLE_YEARS = ["2018", "2019", "2020", "2021", "2023"]
CONSTANTS_PATH = "constants"


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI like gs://bucket/path/to/prefix into (bucket, prefix).
    Ensures the returned prefix always ends with a single '/' (unless empty).
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    remainder = gcs_uri[5:]
    parts = remainder.split("/", 1)
    bucket = parts[0]
    prefix = ""
    if len(parts) == 2:
        prefix = parts[1].strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def should_skip(local_path: Path, blob) -> bool:
    """
    Skip download if a local file exists with the same size.
    """
    try:
        if local_path.exists() and local_path.is_file():
            return local_path.stat().st_size == blob.size
    except Exception:
        pass
    return False


def download_blob(args):
    """Download a single blob from GCS."""
    (blob, base_prefix, dest_dir, skip_existing, chunk_size) = args

    # GCS can have "directory marker" objects ending with '/'
    if blob.name.endswith("/"):
        return f"DIR  : {blob.name} (skipped marker)"

    rel = blob.name[len(base_prefix):].lstrip("/")
    local_path = Path(dest_dir) / rel
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and should_skip(local_path, blob):
        return f"SKIP : {rel} (exists, same size)"

    # Optional: set a chunk size for large files (e.g., 8 MiB)
    if chunk_size:
        blob._chunk_size = chunk_size  # pylint: disable=protected-access

    # Robust retries on transient errors
    retry = Retry(initial=1.0, maximum=30.0, multiplier=2.0, deadline=300.0)
    blob.download_to_filename(str(local_path), retry=retry)
    return f"OK   : {rel}"


def download_gcs_prefix(
    bucket_name: str,
    prefix: str,
    dest_dir: Path,
    creds_path: str,
    skip_existing: bool = True,
    workers: int = 16,
    chunk_size_mb: int = 8,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Download all files from a GCS bucket prefix.

    Args:
        bucket_name: GCS bucket name
        prefix: Prefix/folder path in bucket
        dest_dir: Local destination directory
        creds_path: Path to service account JSON key
        skip_existing: Skip files that already exist locally
        workers: Number of parallel download workers
        chunk_size_mb: Download chunk size in MiB
        verbose: Print progress messages

    Returns:
        Tuple of (successful downloads, errors)
    """
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GCS client
    client = storage.Client.from_service_account_json(creds_path)
    bucket = client.bucket(bucket_name)

    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # List all blobs with the prefix
    if verbose:
        print(f"Listing objects in gs://{bucket_name}/{prefix}")
    blobs_iter = bucket.list_blobs(prefix=prefix)

    # Materialize the list
    blobs = list(blobs_iter)
    if not blobs:
        if verbose:
            print(f"No objects found for prefix: {prefix}")
        return 0, 0

    if verbose:
        print(f"Found {len(blobs)} objects. Starting download to {dest_dir} ...")

    # Prepare work items
    work = []
    chunk_size = chunk_size_mb * 1024 * 1024 if chunk_size_mb > 0 else None
    for b in blobs:
        work.append((b, prefix, dest_dir, skip_existing, chunk_size))

    # Download in parallel
    completed = 0
    errors = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(download_blob, work, chunksize=10):
            if result.startswith("OK"):
                completed += 1
            elif result.startswith("SKIP"):
                if verbose:
                    print(result)
            elif result.startswith("DIR"):
                pass
            else:
                errors += 1

            # Print progress for successful downloads
            if result.startswith("OK") and verbose:
                print(result)

    if verbose:
        print(f"Completed: {completed}, Errors: {errors}, Total: {len(blobs)}")

    return completed, errors


def main():
    p = argparse.ArgumentParser(
        description="Download training data from sewaa-ifs-train GCS bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for specific years
  python download_training_data.py \\
      --creds ~/service-account.json \\
      --dest /mnt/training-data \\
      --years 2018 2019 2020

  # Download everything including constants
  python download_training_data.py \\
      --creds ~/service-account.json \\
      --dest /mnt/training-data \\
      --years 2018 2019 2020 2021 2023 \\
      --download-constants \\
      --skip-existing

  # Custom bucket and paths
  python download_training_data.py \\
      --bucket my-custom-bucket \\
      --creds ~/service-account.json \\
      --dest /mnt/training-data \\
      --custom-paths data/2018 data/2019
        """
    )

    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"GCS bucket name (default: {DEFAULT_BUCKET})",
    )

    p.add_argument(
        "--creds",
        required=True,
        help="Path to JSON service account key file",
    )

    p.add_argument(
        "--dest",
        required=True,
        help="Local destination directory (will create subdirectories for data organization)",
    )

    p.add_argument(
        "--years",
        nargs="+",
        choices=AVAILABLE_YEARS,
        help=f"Years to download (choices: {', '.join(AVAILABLE_YEARS)})",
    )

    p.add_argument(
        "--download-constants",
        action="store_true",
        help="Download constants folder (elevation, land-sea mask, etc.)",
    )

    p.add_argument(
        "--custom-paths",
        nargs="+",
        help="Custom GCS paths to download (overrides --years)",
    )

    p.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 2) * 4),
        help="Number of parallel downloads (default: 16 or 4x CPU count)",
    )

    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally with the same size",
    )

    p.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Download chunk size in MiB (default: 8). Set 0 to use default.",
    )

    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = p.parse_args()

    # Determine what to download
    paths_to_download = []

    if args.custom_paths:
        paths_to_download = args.custom_paths
    else:
        if args.years:
            paths_to_download.extend(args.years)

        if args.download_constants:
            paths_to_download.append(CONSTANTS_PATH)

    if not paths_to_download:
        print("Error: Must specify either --years, --download-constants, or --custom-paths")
        print(f"Available years: {', '.join(AVAILABLE_YEARS)}")
        p.print_help()
        return 1

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("GCS Download Configuration")
        print("=" * 60)
        print(f"Bucket:          gs://{args.bucket}")
        print(f"Paths:           {', '.join(paths_to_download)}")
        print(f"Destination:     {args.dest}")
        print(f"Workers:         {args.workers}")
        print(f"Skip existing:   {args.skip_existing}")
        print(f"Chunk size:      {args.chunk_size_mb} MiB")
        print("=" * 60)
        print()

    # Download each path
    total_completed = 0
    total_errors = 0

    for path in paths_to_download:
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print(f"Downloading: {path}")
            print(f"{'=' * 60}")

        # Determine local destination
        # For year folders, download to dest/{year}
        # For constants, download to dest/constants
        local_dest = Path(args.dest) / path

        try:
            completed, errors = download_gcs_prefix(
                bucket_name=args.bucket,
                prefix=path,
                dest_dir=local_dest,
                creds_path=args.creds,
                skip_existing=args.skip_existing,
                workers=args.workers,
                chunk_size_mb=args.chunk_size_mb,
                verbose=not args.quiet
            )

            total_completed += completed
            total_errors += errors

        except Exception as e:
            print(f"Error downloading {path}: {str(e)}")
            total_errors += 1
            continue

    # Print summary
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print("Download Summary")
        print(f"{'=' * 60}")
        print(f"Total files downloaded:  {total_completed}")
        print(f"Total errors:            {total_errors}")
        print(f"Destination:             {args.dest}")
        print("=" * 60)

        # Print next steps
        print("\nNext Steps:")
        print("1. Update config/data_paths.yaml with the download paths")
        print("2. Verify data structure:")
        print(f"   ls -lh {args.dest}/")
        print("3. Run TFRecords creation:")
        print("   python -c \"from data import write_data; write_data(2018)\"")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(main())

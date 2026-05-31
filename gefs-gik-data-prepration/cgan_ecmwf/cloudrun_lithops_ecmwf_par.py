#!/usr/bin/env python3
"""
Cloud Run Worker for ECMWF Parquet Generation
==============================================

This script runs as a Cloud Run service, receiving HTTP POST requests
from Lithops with date information, running the GIK pipeline, and
uploading parquets to GCS.

Environment Variables:
    PORT: Flask server port (default: 8080)
    AWS_NO_SIGN_REQUEST: Set to 'YES' for anonymous ECMWF S3 access
    GCS_BUCKET: GCS bucket for parquet upload (default: gik-ecmwf-aws-tf)
    GCS_PARQUET_PREFIX: GCS prefix path (default: run_par_ecmwf)
    PARALLEL_WORKERS: Number of parallel workers for Stage 2 (default: 8)
    MAX_MEMBERS: Limit ensemble members for testing (default: None, all 51)

Request Format (POST /process):
    {
        "date": "20260206",  # YYYYMMDD format
        "run": "00",         # Optional, default "00"
        "max_members": null  # Optional, limit members for testing
    }

Response Format:
    {
        "success": true,
        "date": "20260206",
        "run": "00",
        "output_dir": "ecmwf_three_stage_20260206_00z",
        "gcs_path": "gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z",
        "files_count": 51,
        "total_time_seconds": 630.5,
        "message": "Processing completed successfully"
    }

Error Response:
    {
        "success": false,
        "error": "Error message",
        "traceback": "..."
    }

Author: ICPAC GIK Team
Date: 2026-02-08
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add local gik_ecmwf directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
GIK_ECMWF_DIR = SCRIPT_DIR / "gik_ecmwf"
sys.path.insert(0, str(GIK_ECMWF_DIR))

# Environment configuration
PORT = int(os.environ.get('PORT', 8080))
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-ecmwf-aws-tf')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'run_par_ecmwf')
PARALLEL_WORKERS = int(os.environ.get('PARALLEL_WORKERS', 8))
MAX_MEMBERS = os.environ.get('MAX_MEMBERS', None)
if MAX_MEMBERS:
    MAX_MEMBERS = int(MAX_MEMBERS)

# Set anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Template configuration
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz"
LOCAL_TEMPLATE_FILE = "/tmp/gik-fmrc-v2ecmwf_fmrc.tar.gz"

# Create Flask app
app = Flask(__name__)


def download_template_file(url: str, local_path: str) -> bool:
    """Download the pre-built template tar.gz from Hugging Face."""
    if os.path.exists(local_path):
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Template file already exists: {local_path} ({file_size_mb:.1f} MB)")
        return True

    try:
        import requests

        logger.info(f"Downloading template from: {url}")

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                    pct = (downloaded / total_size) * 100
                    logger.info(f"Download progress: {pct:.1f}%")

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Downloaded template: {local_path} ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"Error downloading template: {e}")
        return False


def upload_parquets_to_gcs(
    output_dir: Path,
    date_str: str,
    run: str
) -> tuple[str, int]:
    """
    Upload final parquet files to GCS.

    Returns (gcs_path, files_count) tuple.
    """
    try:
        import gcsfs

        logger.info(f"Setting up GCS upload to bucket: {GCS_BUCKET}")

        # Use Application Default Credentials (ADC)
        fs = gcsfs.GCSFileSystem()

        # Build GCS path
        gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{date_str}_{run}z"
        logger.info(f"GCS destination: gs://{gcs_path}")

        # Find all *_final.parquet files
        parquet_files = list(output_dir.glob("*_final.parquet"))

        if not parquet_files:
            logger.warning(f"No *_final.parquet files found in {output_dir}")
            return None, 0

        logger.info(f"Found {len(parquet_files)} parquet files to upload")

        # Upload each file
        uploaded_count = 0
        total_size_mb = 0

        for pf in parquet_files:
            gcs_file_path = f"{gcs_path}/{pf.name}"
            try:
                fs.put(str(pf), gcs_file_path)
                size_kb = pf.stat().st_size / 1024
                total_size_mb += size_kb / 1024
                logger.info(f"Uploaded: {pf.name} ({size_kb:.1f} KB)")
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {pf.name}: {e}")

        logger.info(f"Uploaded {uploaded_count}/{len(parquet_files)} files ({total_size_mb:.2f} MB total)")

        return f"gs://{gcs_path}", uploaded_count

    except ImportError:
        logger.error("gcsfs not installed")
        return None, 0
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        logger.error(traceback.format_exc())
        return None, 0


def process_date(date_str: str, run: str = "00", max_members: int = None) -> dict:
    """
    Run the GIK three-stage pipeline for a single date.

    Args:
        date_str: Target date in YYYYMMDD format
        run: Model run hour (00 or 12)
        max_members: Optional limit on ensemble members

    Returns:
        Dictionary with processing results
    """
    start_time = time.time()

    try:
        logger.info(f"="*70)
        logger.info(f"Starting ECMWF GIK processing")
        logger.info(f"Date: {date_str}, Run: {run}z")
        logger.info(f"Max members: {max_members if max_members else 'all (51)'}")
        logger.info(f"Parallel workers: {PARALLEL_WORKERS}")
        logger.info(f"="*70)

        # Step 1: Download template
        if not download_template_file(TEMPLATE_URL, LOCAL_TEMPLATE_FILE):
            return {
                "success": False,
                "error": "Failed to download template file"
            }

        # Step 2: Run three-stage pipeline
        from ecmwf_three_stage_multidate import process_single_date

        logger.info("Running three-stage pipeline with template fast-path")

        success, output_dir = process_single_date(
            date_str=date_str,
            run=run,
            max_members=max_members,
            use_local_template=True,
            local_template_path=LOCAL_TEMPLATE_FILE,
            skip_grib_scan=True,  # Use template fast-path
            parallel_workers=PARALLEL_WORKERS
        )

        if not success:
            return {
                "success": False,
                "error": "Three-stage pipeline failed",
                "output_dir": str(output_dir) if output_dir else None
            }

        logger.info(f"Pipeline completed. Output: {output_dir}")

        # Step 3: Upload to GCS
        output_path = Path(output_dir)
        gcs_path, files_count = upload_parquets_to_gcs(
            output_dir=output_path,
            date_str=date_str,
            run=run
        )

        total_time = time.time() - start_time

        # Build response
        result = {
            "success": True,
            "date": date_str,
            "run": run,
            "output_dir": str(output_dir),
            "gcs_path": gcs_path,
            "files_count": files_count,
            "total_time_seconds": round(total_time, 2),
            "message": "Processing completed successfully"
        }

        logger.info(f"Processing completed in {total_time:.1f} seconds")
        logger.info(f"GCS path: {gcs_path}")
        logger.info(f"Files uploaded: {files_count}")

        return result

    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        tb = traceback.format_exc()

        logger.error(f"Processing failed: {error_msg}")
        logger.error(tb)

        return {
            "success": False,
            "date": date_str,
            "run": run,
            "error": error_msg,
            "traceback": tb,
            "total_time_seconds": round(total_time, 2)
        }


# ==============================================================================
# Flask Routes
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "ecmwf-parquet-worker"}), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information."""
    return jsonify({
        "name": "ECMWF Parquet Worker",
        "version": "1.0.0",
        "description": "Cloud Run worker for GIK-based ECMWF parquet generation",
        "endpoints": {
            "/health": "Health check",
            "/process": "Process ECMWF date (POST)"
        },
        "configuration": {
            "gcs_bucket": GCS_BUCKET,
            "gcs_prefix": GCS_PARQUET_PREFIX,
            "parallel_workers": PARALLEL_WORKERS,
            "max_members": MAX_MEMBERS if MAX_MEMBERS else "all"
        }
    }), 200


@app.route('/process', methods=['POST'])
def process():
    """
    Process a single ECMWF date.

    Expected JSON payload:
        {
            "date": "20260206",       # Required: YYYYMMDD
            "run": "00",              # Optional: default "00"
            "max_members": null       # Optional: limit members
        }
    """
    try:
        # Parse request
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.get_json()

        # Validate required fields
        if 'date' not in data:
            return jsonify({"success": False, "error": "Missing 'date' field"}), 400

        date_str = str(data['date'])
        run = str(data.get('run', '00'))
        max_members = data.get('max_members', MAX_MEMBERS)

        # Validate date format
        try:
            datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid date format: {date_str}. Expected YYYYMMDD"
            }), 400

        # Process date
        logger.info(f"Received processing request for date: {date_str}, run: {run}")

        result = process_date(date_str, run, max_members)

        # Return result with appropriate status code
        status_code = 200 if result['success'] else 500

        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Request handling error: {e}")
        logger.error(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    logger.info(f"Starting ECMWF Parquet Worker on port {PORT}")
    logger.info(f"GCS bucket: {GCS_BUCKET}")
    logger.info(f"GCS prefix: {GCS_PARQUET_PREFIX}")
    logger.info(f"Parallel workers: {PARALLEL_WORKERS}")

    # Run Flask app
    app.run(host='0.0.0.0', port=PORT, debug=False)

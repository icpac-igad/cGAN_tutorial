# Scripts for cGAN Data Management

This directory contains utility scripts for downloading and managing training data for the cGAN precipitation downscaling model.

## Scripts Overview

### 1. `download_training_data.py`
Downloads forecast, truth, and constant files from the `sewaa-ifs-train` GCS bucket.

**Features:**
- Parallel downloads with configurable workers
- Skip existing files (resume capability)
- Automatic retry on transient errors
- Progress reporting
- Download specific years or custom paths

### 2. `setup_data_config.py`
Configures the `config/data_paths.yaml` file to point to downloaded training data.

**Features:**
- Interactive or command-line mode
- Automatic path configuration
- Backup of existing configuration
- Optional update of `local_config.yaml`

### 3. Other Scripts
- `convert_zarr_to_netcdf.py` - Convert Zarr format to NetCDF
- `generate_fcst_norm.py` - Generate forecast normalization constants

## Quick Start

### Step 1: Download Training Data

Download data for specific years from the GCS bucket:

```bash
# Download years 2018, 2019, 2020 and constants
python download_training_data.py \
    --creds ~/service-account.json \
    --dest /mnt/training-data \
    --years 2018 2019 2020 \
    --download-constants \
    --skip-existing \
    --workers 16
```

**Arguments:**
- `--creds`: Path to GCS service account JSON key (required)
- `--dest`: Local destination directory (required)
- `--years`: Years to download (choices: 2018, 2019, 2020, 2021, 2023)
- `--download-constants`: Also download constants folder
- `--skip-existing`: Skip files that already exist locally
- `--workers`: Number of parallel download workers (default: 16)
- `--bucket`: GCS bucket name (default: sewaa-ifs-train)

**Expected download structure:**
```
/mnt/training-data/
├── 2018/
│   ├── file1.nc
│   ├── file2.nc
│   └── ...
├── 2019/
├── 2020/
└── constants/
    ├── elevation.nc
    ├── lsm.nc
    └── ...
```

### Step 2: Configure Data Paths

After downloading, configure the cGAN project to use the downloaded data:

```bash
# Interactive mode
python setup_data_config.py

# Command line mode
python setup_data_config.py \
    --config-name VM_SESSION \
    --base-path /mnt/training-data \
    --update-local-config
```

**Arguments:**
- `--config-name`: Name for this configuration (e.g., VM_SESSION, LOCAL_DATA)
- `--base-path`: Base directory where data was downloaded
- `--forecast-path`: Custom forecast path (default: base-path)
- `--truth-path`: Custom truth path (default: base-path/TRUTH)
- `--constants-path`: Custom constants path (default: base-path/constants)
- `--tfrecords-path`: Custom TFRecords path (default: base-path/tfrecords)
- `--update-local-config`: Also update local_config.yaml to use this configuration
- `--dry-run`: Print configuration without writing files

### Step 3: Verify Configuration

```bash
# Test that configuration is correctly set
python -c "from config import get_data_paths; import pprint; pprint.pprint(get_data_paths())"
```

Expected output:
```python
{'GENERAL': {'CONSTANTS_PATH': '/mnt/training-data/constants/',
             'FORECAST_PATH': '/mnt/training-data/',
             'TRUTH_PATH': '/mnt/training-data/TRUTH/'},
 'TFRecords': {'tfrecords_path': '/mnt/training-data/tfrecords/'}}
```

## Detailed Usage Examples

### Example 1: Download All Available Data

```bash
python download_training_data.py \
    --creds ~/gcs-key.json \
    --dest /data/cgan-training \
    --years 2018 2019 2020 2021 2023 \
    --download-constants \
    --skip-existing
```

### Example 2: Download Only Constants

```bash
python download_training_data.py \
    --creds ~/gcs-key.json \
    --dest /data/cgan-training \
    --download-constants
```

### Example 3: Download from Custom Bucket/Paths

```bash
python download_training_data.py \
    --bucket my-custom-bucket \
    --creds ~/gcs-key.json \
    --dest /data/cgan-training \
    --custom-paths data/forecast/2018 data/truth/2018 data/constants
```

### Example 4: Resume Interrupted Download

The `--skip-existing` flag allows you to resume interrupted downloads:

```bash
# If download was interrupted, re-run with same command
python download_training_data.py \
    --creds ~/gcs-key.json \
    --dest /mnt/training-data \
    --years 2018 2019 \
    --skip-existing  # Will skip already downloaded files
```

### Example 5: Configure Custom Paths

```bash
# If your data has a different structure
python setup_data_config.py \
    --config-name MY_MACHINE \
    --base-path /data/cgan \
    --forecast-path /data/cgan/forecasts \
    --truth-path /data/cgan/observations \
    --constants-path /data/cgan/static \
    --tfrecords-path /data/cgan/tfrecords \
    --update-local-config
```

## Integration with cGAN Workflow

After downloading and configuring data paths, you can proceed with TFRecords creation:

### 1. Generate Forecast Normalization

```bash
cd /path/to/cGAN_tutorial
python -c "from data import gen_fcst_norm; gen_fcst_norm(year='2018')"
```

This creates `FCSTNorm2018.pkl` in the CONSTANTS_PATH directory.

### 2. Create TFRecords

```bash
cd /path/to/cGAN_tutorial

# Create TFRecords for year 2018
python << 'EOFPYTHON'
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from data import write_data
write_data(2018)
EOFPYTHON
```

Or use the notebook:
```bash
jupyter notebook example_notebooks/create_tfrecords.ipynb
```

### 3. Verify TFRecords

```bash
python << 'EOFPYTHON'
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from config import get_data_paths
from data import _parse_batch

data_paths = get_data_paths()
tfrecords_path = data_paths["TFRecords"]["tfrecords_path"]

# List TFRecord files
import glob
files = glob.glob(f"{tfrecords_path}/*.tfrecords")
print(f"Found {len(files)} TFRecord files")

# Test loading a sample
if files:
    dataset = tf.data.TFRecordDataset(files[0], compression_type='GZIP')
    dataset = dataset.map(lambda x: _parse_batch(x,
                                                   insize=(128,128,56),
                                                   consize=(128,128,2),
                                                   outsize=(128,128,1)))
    for inputs, outputs in dataset.take(1):
        print(f"lo_res_inputs: {inputs['lo_res_inputs'].shape}")
        print(f"hi_res_inputs: {inputs['hi_res_inputs'].shape}")
        print(f"output: {outputs['output'].shape}")
EOFPYTHON
```

## File Organization

After running the scripts, your data should be organized as:

```
dest_directory/
├── 2018/               # Downloaded year folders
│   ├── *.nc           # NetCDF files for this year
│   └── ...
├── 2019/
├── 2020/
├── 2021/
├── 2023/
├── constants/         # Static fields
│   ├── elevation.nc
│   ├── lsm.nc
│   └── FCSTNorm2018.pkl  # Created by gen_fcst_norm()
├── TRUTH/             # If truth data is separate
│   └── *.nc
└── tfrecords/         # Created by write_data()
    ├── 2018_30.0.tfrecords
    ├── 2018_30.1.tfrecords
    └── ...
```

## Configuration Files

### config/data_paths.yaml

Contains path configurations for different machines/environments:

```yaml
VM_SESSION:
  GENERAL:
    TRUTH_PATH: '/mnt/training-data/TRUTH/'
    FORECAST_PATH: '/mnt/training-data/'
    CONSTANTS_PATH: '/mnt/training-data/constants/'
  TFRecords:
    tfrecords_path: '/mnt/training-data/tfrecords/'

LOCAL_DATA:
  GENERAL:
    TRUTH_PATH: '/data/cgan/TRUTH/'
    FORECAST_PATH: '/data/cgan/'
    CONSTANTS_PATH: '/data/cgan/constants/'
  TFRecords:
    tfrecords_path: '/data/cgan/tfrecords/'
```

### config/local_config.yaml

Specifies which configuration to use:

```yaml
data_paths: "VM_SESSION"  # References a key in data_paths.yaml
gpu_mem_incr: True
use_gpu: True
disable_tf32: False
```

## Troubleshooting

### Issue: Authentication Error

```bash
# Verify GCS credentials
gcloud auth application-default login

# Or verify service account key
python -c "from google.cloud import storage; client = storage.Client.from_service_account_json('~/service-account.json'); print('Auth OK')"
```

### Issue: Download Fails

```bash
# Check bucket access
gsutil ls gs://sewaa-ifs-train/

# Test with single file
gsutil cp gs://sewaa-ifs-train/constants/elevation.nc /tmp/test.nc

# Check network connectivity
ping -c 3 storage.googleapis.com
```

### Issue: Configuration Not Found

```bash
# Verify config files exist
ls -l config/data_paths.yaml
ls -l config/local_config.yaml

# Check Python can find config module
python -c "from config import get_data_paths; print('Config OK')"
```

### Issue: Path Mismatch

If the downloaded data structure doesn't match expectations:

```bash
# Check actual structure
tree -L 2 /mnt/training-data/

# Manually configure paths
python setup_data_config.py  # Use interactive mode
```

## Performance Tips

### Optimize Download Speed

```bash
# Increase workers for faster downloads (if bandwidth allows)
python download_training_data.py \
    --workers 32 \
    --chunk-size-mb 16 \
    ...
```

### Reduce Storage Usage

```bash
# Download only specific years needed for training
python download_training_data.py \
    --years 2018 2019 \
    ...

# Don't download constants if already available
python download_training_data.py \
    --years 2018 2019 \
    # omit --download-constants
```

### Resume Interrupted Downloads

Always use `--skip-existing` to avoid re-downloading files:

```bash
python download_training_data.py \
    --skip-existing \
    ...
```

## Advanced Usage

### Custom GCS Bucket Structure

If your data is in a different bucket or structure:

```bash
python download_training_data.py \
    --bucket my-bucket \
    --custom-paths \
        forecasts/ifs/2018 \
        forecasts/ifs/2019 \
        observations/imerg/2018 \
        static/constants \
    --creds ~/key.json \
    --dest /data/cgan
```

### Multiple Environments

Configure different environments for different machines:

```bash
# On VM
python setup_data_config.py \
    --config-name VM_SESSION \
    --base-path /mnt/training-data \
    --update-local-config

# On local machine
python setup_data_config.py \
    --config-name LOCAL_DEV \
    --base-path /home/user/data/cgan \
    --update-local-config
```

### Parallel Processing

Download multiple paths simultaneously using multiple terminal sessions:

```bash
# Terminal 1: Download 2018-2020
python download_training_data.py --years 2018 2019 2020 ... &

# Terminal 2: Download 2021-2023
python download_training_data.py --years 2021 2023 ... &

# Wait for both to complete
wait
```

## Related Documentation

- `../docs/tfrecords_creation_workflow.md` - Full workflow documentation
- `../example_notebooks/create_tfrecords.ipynb` - Interactive notebook
- `../config/README.md` - Configuration system documentation (if exists)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main documentation in `docs/`
3. Check the example notebooks in `example_notebooks/`
4. Open an issue on the repository

## Version History

- v1.0 (2025-11-19): Initial release with download and configuration scripts

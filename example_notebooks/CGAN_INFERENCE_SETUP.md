# cGAN Inference Setup and Technical Debt Documentation

## Overview

This document details the required files, setup, and technical debt for running cGAN inference on GEFS (Global Ensemble Forecast System) data using the provided workflow.

## Inference Code Entry Point

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Required for TensorFlow >= 2.16.0

import sys
sys.path.insert(1,"../scripts/")
import forecast_gfs
import importlib
importlib.reload(forecast_gfs)

forecast_gfs.make_fcst()
```

## Required Files and Directory Structure

### 1. Core Script Files

```
cGAN_tutorial/
├── scripts/
│   └── forecast_gfs.py           # Main inference script
├── config/
│   ├── forecast_gfs.yaml         # Inference configuration (in ../config/)
│   ├── data_paths.yaml           # Data path configuration
│   ├── local_config.yaml         # Local environment settings
│   └── downscaling_factor.yaml   # Model architecture config
├── data/
│   └── data_gefs.py              # Data loading utilities
├── model/
│   ├── gan.py                    # GAN model definitions
│   ├── models.py                 # Generator/Discriminator architectures
│   ├── noise.py                  # Noise generation for ensemble members
│   └── blocks.py, layers.py      # Network building blocks
└── setupmodel.py                 # Model initialization utilities
```

### 2. Configuration Files

#### `forecast_gfs.yaml` (Main inference config)
```yaml
MODEL:
    folder: "/path/to/trained/model/"    # Contains setup_params.yaml and models/
    checkpoint: 345600                    # Checkpoint number to load

INPUT:
    folder: "/path/to/input/netcdf/"     # Input forecast data folder
    dates: ["2024-04-20"]                # List of forecast dates
    start_hour: 30                        # First forecast hour
    end_hour: 54                          # Last forecast hour

OUTPUT:
    folder: "/path/to/output/"           # Output folder for predictions
    ensemble_members: 50                  # Number of ensemble members to generate
```

#### `data_paths.yaml`
Must contain paths for the environment specified in `local_config.yaml`:
```yaml
ICPAC_CLOUD:  # or your environment name
    GENERAL:
        TRUTH_PATH: "/path/to/truth/data/"
        FORECAST_PATH: "/path/to/forecast/data/"
        CONSTANTS_PATH_GEFS: "/path/to/constants/"
    TFRecords:
        tfrecords_path: "/path/to/tfrecords/"
```

#### `local_config.yaml`
```yaml
data_paths: "ICPAC_CLOUD"      # Must match key in data_paths.yaml
gpu_mem_incr: True              # Incremental GPU memory allocation
use_gpu: True                   # Enable GPU usage
disable_tf32: False             # TensorFloat-32 setting
```

#### `downscaling_factor.yaml`
```yaml
downscaling_factor: 1
steps: [1]  # Must multiply to downscaling_factor
```

### 3. Required Data Files

#### Input Forecast Data (NetCDF)
Located in `INPUT.folder/YYYY/`:
```
{field}_YYYY.nc  # One file per field per year
```

Fields required (from `data_gefs.py:25`):
- `cape` - Convective Available Potential Energy
- `pres` - Pressure
- `pwat` - Precipitable Water
- `tmp` - Temperature
- `ugrd` - U-component of wind
- `vgrd` - V-component of wind
- `msl` - Mean Sea Level Pressure
- `apcp` - Accumulated Precipitation

#### Constants Data
Located in `CONSTANTS_PATH_GEFS`:
- `elev.nc` - Elevation/orography data
- `lsm.nc` - Land-sea mask
- `FCSTNorm2018.pkl` - Normalization statistics (generated via `gen_fcst_norm()`)

#### Pre-trained Model
Located in `MODEL.folder`:
```
setup_params.yaml               # Model architecture config
models/
    └── gen_weights-XXXXXXX.h5  # Generator weights checkpoint
```

### 4. Python Dependencies

**Core Dependencies:**
- TensorFlow (2.x with Keras support)
- NumPy
- xarray
- netCDF4
- cftime
- h5py (for IMERG data handling)
- PyYAML
- xesmf (for regridding operations)

**Environment Variable:**
```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TensorFlow >= 2.16.0
```

## Technical Debt and Issues

### 1. **Hard-coded Geographic Region** 🔴 HIGH PRIORITY
**Location:** `forecast_gfs.py:39-40`, `data_gefs.py:30-36`

```python
# Hard-coded ICPAC region coordinates
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)
```

**Issues:**
- Geographic bounds are hard-coded for ICPAC region
- Not generalizable to other regions
- Coordinate information should be read from input files

**Fix:**
- Extract lat/lon from input NetCDF files
- Make region configurable via YAML
- Add validation for coordinate consistency

### 2. **Duplicate Normalization Logic** 🔴 HIGH PRIORITY
**Location:** `forecast_gfs.py:224-252` vs `data_gefs.py:273-299`

**Issues:**
- Same normalization logic duplicated in two places
- Difficult to maintain consistency
- Violates DRY (Don't Repeat Yourself) principle

**Fix:**
```python
# Consolidate into shared utility function
from data.data_gefs import normalize_forecast_field

def load_and_normalize_field(field, nc_file, in_time_idx, fcst_norm):
    return normalize_forecast_field(field, nc_file, in_time_idx, fcst_norm)
```

### 3. **Path Management Issues** 🟡 MEDIUM PRIORITY
**Location:** `forecast_gfs.py:50`, `data_gefs.py:16`

```python
sys.path.insert(1,"../")  # Fragile relative path manipulation
```

**Issues:**
- Fragile relative paths depend on execution location
- Makes code non-portable
- Breaks when run from different directories

**Fix:**
- Use proper Python package structure
- Install as editable package: `pip install -e .`
- Use absolute imports with package namespace

### 4. **Configuration File Redundancy** 🟡 MEDIUM PRIORITY
**Location:** `example_notebooks/forecast_gfs.yaml` vs `config/forecast_gfs.yaml`

**Issues:**
- Two versions of same config file in different locations
- Confusion about which one is used
- Script looks in `../config/` but there's one in current dir

**Fix:**
- Use single config location (preferably `config/`)
- Document config precedence clearly
- Add config validation at startup

### 5. **Environment Variable Requirement** 🟡 MEDIUM PRIORITY
**Location:** Entry point code

```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

**Issues:**
- Must be set before TensorFlow import
- Easy to forget or do incorrectly
- Version-dependent workaround

**Fix:**
- Document TensorFlow version requirements clearly
- Consider pinning TensorFlow < 2.16 if legacy Keras is required
- Add startup validation to check environment

### 6. **Magic Numbers and Assumptions** 🟡 MEDIUM PRIORITY
**Location:** `forecast_gfs.py:217`, `data_gefs.py:28`

```python
nc_file.isel({"step":[in_time_idx-5,in_time_idx-4]})  # What are -5, -4?
HOURS = 6  # Hard-coded temporal resolution
```

**Issues:**
- Magic numbers without explanation
- Assumptions about data structure not validated
- Time indexing logic is unclear

**Fix:**
- Add comments explaining index offsets
- Validate input file structure matches assumptions
- Make temporal resolution configurable

### 7. **Error Handling Gaps** 🟡 MEDIUM PRIORITY
**Location:** Throughout `forecast_gfs.py`

**Issues:**
- No validation of input file existence before processing
- No checks for data shape/dimension compatibility
- Silent failures possible with `np.maximum(data, 0.0)`

**Fix:**
```python
# Add validation
def validate_input_data(nc_file, expected_shape, field_name):
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"Input file not found: {nc_file}")
    # ... dimension checks
```

### 8. **Mixed File Format Support** 🟢 LOW PRIORITY
**Location:** `forecast_gfs.py:209-215`

```python
# Commented out Zarr, using NetCDF
# input_file = f"{field}_{d.year}.zarr"
# nc_in_path = os.path.join(input_folder_year, input_file)
# nc_file = xr.open_zarr(nc_in_path)
input_file = f"{field}_{d.year}.nc"
```

**Issues:**
- Dead code (commented Zarr support)
- Unclear which format should be used
- Maintenance burden

**Fix:**
- Remove dead code
- Document required file format clearly
- If both formats needed, make it a config option

### 9. **Output Directory Structure** 🟢 LOW PRIORITY
**Location:** `forecast_gfs.py:181`

```python
output_folder_year = output_folder+f"test/{d.year}/"
```

**Issues:**
- Hard-coded "test" subdirectory
- Unclear why "test" is in path
- Should be configurable

**Fix:**
- Add output_subdirectory config parameter
- Or remove "test" if not needed

### 10. **Model Architecture Assumptions** 🟢 LOW PRIORITY
**Location:** `forecast_gfs.py:87`

```python
assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"
```

**Issues:**
- Script only works with GAN mode
- Could be extended to support other architectures
- Limits reusability

**Fix:**
- Extend support to VAE-GAN and deterministic models
- Or clearly document GAN-only limitation in docs

## Recommended Improvements

### Priority 1: Critical Fixes
1. **Consolidate normalization logic** into shared utilities
2. **Remove hard-coded coordinates** - read from input files
3. **Fix path management** - use proper package structure
4. **Add input validation** - check files exist and have expected structure

### Priority 2: Code Quality
5. **Clean up configuration** - single source of truth for configs
6. **Document magic numbers** - explain all index offsets and constants
7. **Improve error messages** - helpful errors when things go wrong
8. **Add logging** - track progress and debug issues

### Priority 3: Future Enhancements
9. **Support multiple regions** - make geographic area configurable
10. **Parallel processing** - process multiple dates/ensemble members concurrently
11. **Output format options** - support Zarr, HDF5 in addition to NetCDF
12. **Model agnostic inference** - support VAE-GAN and deterministic models

## Setup Checklist

- [ ] Install Python dependencies (TensorFlow, xarray, netCDF4, etc.)
- [ ] Set `TF_USE_LEGACY_KERAS=1` environment variable
- [ ] Configure `local_config.yaml` with your environment name
- [ ] Update `data_paths.yaml` with correct paths for your environment
- [ ] Configure `forecast_gfs.yaml` with model path, dates, and output location
- [ ] Ensure trained model exists at specified path with correct checkpoint
- [ ] Verify input NetCDF files exist for all required fields
- [ ] Generate normalization statistics (`FCSTNorm2018.pkl`) if not present
- [ ] Verify constants files exist (elevation, land-sea mask)
- [ ] Create output directory structure
- [ ] Test with single date before batch processing

## Usage Example

```python
# 1. Set environment variable (must be before imports)
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 2. Import and reload module
import sys
sys.path.insert(1, "../scripts/")
import forecast_gfs
import importlib
importlib.reload(forecast_gfs)

# 3. Run inference (uses config from ../config/forecast_gfs.yaml)
forecast_gfs.make_fcst()
```

## Output

The script generates NetCDF files:
```
OUTPUT.folder/test/YYYY/GAN_YYYYMMDD.nc
```

Structure:
- Dimensions: `(time, member, valid_time, latitude, longitude)`
- Variable: `precipitation` in mm/h
- Ensemble members: Configured number of stochastic realizations
- Valid times: From `start_hour` to `end_hour` at 6-hour intervals

## Troubleshooting

**Import errors:**
- Verify `sys.path.insert()` is correct relative to execution location
- Consider installing package in editable mode instead

**Missing normalization file:**
- Run `generate_fcst_norm.py` to create `FCSTNorm2018.pkl`
- Ensure CONSTANTS_PATH_GEFS is correctly configured

**GPU memory issues:**
- Set `gpu_mem_incr: True` in `local_config.yaml`
- Reduce `ensemble_members` in forecast config
- Process fewer dates at once

**File not found errors:**
- Check all paths in `data_paths.yaml` are absolute and correct
- Verify input files follow naming convention: `{field}_YYYY.nc`
- Ensure year folder exists in input directory

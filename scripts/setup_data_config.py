#!/usr/bin/env python3
"""
Configure data paths for cGAN training after downloading from GCS.

This script helps set up the config/data_paths.yaml file to point to
downloaded training data.

Usage:
    # Interactive mode
    python setup_data_config.py

    # Command line mode
    python setup_data_config.py \\
        --config-name MY_MACHINE \\
        --base-path /mnt/training-data \\
        --forecast-subdir 2018,2019,2020 \\
        --truth-subdir TRUTH \\
        --constants-subdir constants \\
        --tfrecords-subdir tfrecords

Example:
    python setup_data_config.py \\
        --config-name VM_SESSION \\
        --base-path /mnt/training-data
"""

import argparse
import os
import sys
from pathlib import Path


def get_repo_root():
    """Find the repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root (no 'config' directory)")


def read_data_paths_yaml(config_path):
    """Read existing data_paths.yaml file."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def write_data_paths_yaml(config_path, data):
    """Write data_paths.yaml file."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_config_entry(
    config_name,
    base_path,
    forecast_path=None,
    truth_path=None,
    constants_path=None,
    tfrecords_path=None
):
    """
    Generate a configuration entry for data_paths.yaml.

    Args:
        config_name: Name of the configuration (e.g., "VM_SESSION")
        base_path: Base directory where data is stored
        forecast_path: Path to forecast data (default: {base_path})
        truth_path: Path to truth data (default: {base_path}/TRUTH)
        constants_path: Path to constants (default: {base_path}/constants)
        tfrecords_path: Path to TFRecords (default: {base_path}/tfrecords)

    Returns:
        Dictionary with configuration entry
    """
    base_path = str(Path(base_path).resolve())

    # Set defaults
    if forecast_path is None:
        forecast_path = base_path
    if truth_path is None:
        truth_path = os.path.join(base_path, "TRUTH")
    if constants_path is None:
        constants_path = os.path.join(base_path, "constants")
    if tfrecords_path is None:
        tfrecords_path = os.path.join(base_path, "tfrecords")

    # Ensure paths end with /
    forecast_path = forecast_path.rstrip("/") + "/"
    truth_path = truth_path.rstrip("/") + "/"
    constants_path = constants_path.rstrip("/") + "/"
    tfrecords_path = tfrecords_path.rstrip("/") + "/"

    return {
        config_name: {
            "GENERAL": {
                "TRUTH_PATH": truth_path,
                "FORECAST_PATH": forecast_path,
                "CONSTANTS_PATH": constants_path,
            },
            "TFRecords": {
                "tfrecords_path": tfrecords_path,
            }
        }
    }


def interactive_setup():
    """Interactive configuration setup."""
    print("=" * 60)
    print("Interactive Data Paths Configuration")
    print("=" * 60)
    print()

    # Get configuration name
    config_name = input("Configuration name (e.g., VM_SESSION, LOCAL_DATA): ").strip()
    if not config_name:
        print("Error: Configuration name cannot be empty")
        return None

    # Get base path
    base_path = input("Base path where data is stored: ").strip()
    if not base_path:
        print("Error: Base path cannot be empty")
        return None

    base_path = Path(base_path).expanduser().resolve()

    # Ask for custom paths
    print("\nCustomize paths? (leave empty to use defaults)")

    forecast_path = input(f"Forecast path (default: {base_path}): ").strip()
    if not forecast_path:
        forecast_path = str(base_path)

    truth_path = input(f"Truth path (default: {base_path}/TRUTH): ").strip()
    if not truth_path:
        truth_path = str(base_path / "TRUTH")

    constants_path = input(f"Constants path (default: {base_path}/constants): ").strip()
    if not constants_path:
        constants_path = str(base_path / "constants")

    tfrecords_path = input(f"TFRecords path (default: {base_path}/tfrecords): ").strip()
    if not tfrecords_path:
        tfrecords_path = str(base_path / "tfrecords")

    return generate_config_entry(
        config_name,
        str(base_path),
        forecast_path=forecast_path,
        truth_path=truth_path,
        constants_path=constants_path,
        tfrecords_path=tfrecords_path
    )


def main():
    p = argparse.ArgumentParser(
        description="Configure data paths for cGAN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python setup_data_config.py

  # Command line mode with defaults
  python setup_data_config.py \\
      --config-name VM_SESSION \\
      --base-path /mnt/training-data

  # Command line mode with custom paths
  python setup_data_config.py \\
      --config-name VM_SESSION \\
      --base-path /mnt/training-data \\
      --forecast-path /mnt/training-data/forecast \\
      --truth-path /mnt/training-data/truth \\
      --constants-path /mnt/training-data/constants \\
      --tfrecords-path /mnt/training-data/tfrecords
        """
    )

    p.add_argument(
        "--config-name",
        help="Name for this configuration (e.g., VM_SESSION, LOCAL_DATA)",
    )

    p.add_argument(
        "--base-path",
        help="Base directory where data is stored",
    )

    p.add_argument(
        "--forecast-path",
        help="Path to forecast data (default: base-path)",
    )

    p.add_argument(
        "--truth-path",
        help="Path to truth data (default: base-path/TRUTH)",
    )

    p.add_argument(
        "--constants-path",
        help="Path to constants (default: base-path/constants)",
    )

    p.add_argument(
        "--tfrecords-path",
        help="Path to TFRecords (default: base-path/tfrecords)",
    )

    p.add_argument(
        "--update-local-config",
        action="store_true",
        help="Also update config/local_config.yaml to use this configuration",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without writing files",
    )

    args = p.parse_args()

    # Find repository root
    try:
        repo_root = get_repo_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    config_path = repo_root / "config" / "data_paths.yaml"

    # Generate configuration
    if args.config_name and args.base_path:
        # Command line mode
        new_config = generate_config_entry(
            args.config_name,
            args.base_path,
            forecast_path=args.forecast_path,
            truth_path=args.truth_path,
            constants_path=args.constants_path,
            tfrecords_path=args.tfrecords_path
        )
    else:
        # Interactive mode
        new_config = interactive_setup()
        if new_config is None:
            sys.exit(1)

    # Load existing config
    existing_config = read_data_paths_yaml(config_path)

    # Merge configurations
    existing_config.update(new_config)

    # Print configuration
    config_name = list(new_config.keys())[0]
    print("\n" + "=" * 60)
    print("Generated Configuration")
    print("=" * 60)
    print(f"\n{config_name}:")
    print(f"  GENERAL:")
    print(f"    TRUTH_PATH: {new_config[config_name]['GENERAL']['TRUTH_PATH']}")
    print(f"    FORECAST_PATH: {new_config[config_name]['GENERAL']['FORECAST_PATH']}")
    print(f"    CONSTANTS_PATH: {new_config[config_name]['GENERAL']['CONSTANTS_PATH']}")
    print(f"  TFRecords:")
    print(f"    tfrecords_path: {new_config[config_name]['TFRecords']['tfrecords_path']}")
    print()

    # Write or dry-run
    if args.dry_run:
        print("Dry run mode - no files written")
        print(f"\nWould write to: {config_path}")
    else:
        # Backup existing config
        if config_path.exists():
            backup_path = config_path.with_suffix(".yaml.backup")
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"Backed up existing config to: {backup_path}")

        # Write new config
        write_data_paths_yaml(config_path, existing_config)
        print(f"Updated: {config_path}")

        # Update local_config.yaml if requested
        if args.update_local_config:
            local_config_path = repo_root / "config" / "local_config.yaml"
            if local_config_path.exists():
                # Read existing local config
                with open(local_config_path, "r") as f:
                    import yaml
                    local_config = yaml.safe_load(f) or {}

                # Update data_paths setting
                local_config["data_paths"] = config_name

                # Write back
                with open(local_config_path, "w") as f:
                    yaml.dump(local_config, f, default_flow_style=False, sort_keys=False)

                print(f"Updated local_config.yaml to use '{config_name}'")

    # Print next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print(f"1. Verify data exists at configured paths:")
    print(f"   ls {new_config[config_name]['GENERAL']['FORECAST_PATH']}")
    print(f"   ls {new_config[config_name]['GENERAL']['TRUTH_PATH']}")
    print(f"   ls {new_config[config_name]['GENERAL']['CONSTANTS_PATH']}")
    print()
    print("2. Update config/local_config.yaml to use this configuration:")
    print(f"   data_paths: \"{config_name}\"")
    print()
    print("3. Test configuration:")
    print("   python -c \"from config import get_data_paths; import pprint; pprint.pprint(get_data_paths())\"")
    print()
    print("4. Generate forecast normalization:")
    print("   python -c \"from data import gen_fcst_norm; gen_fcst_norm(year='2018')\"")
    print()
    print("5. Create TFRecords:")
    print("   python -c \"from data import write_data; write_data(2018)\"")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

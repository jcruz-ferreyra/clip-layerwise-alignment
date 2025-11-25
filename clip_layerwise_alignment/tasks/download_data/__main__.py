# tasks/download_data/__main__.py

from pathlib import Path

from clip_layerwise_alignment.config import LOCAL_DATA_DIR, DRIVE_DATA_DIR
from clip_layerwise_alignment.utils import check_missing_keys, load_config, setup_logging

# Setup logging
script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

# Import task components
from clip_layerwise_alignment.tasks.download_data import (
    DownloadDataContext,
    download_data,
)

logger.info("=" * 80)
logger.info("Starting download_data task")
logger.info("=" * 80)

# Load config
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"
logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

# Validate config
required_keys = ["datasets"]
check_missing_keys(required_keys, script_config)

# Parse config
DATASETS = script_config["datasets"]
OUTPUT_STORAGE = script_config.get("output_storage", "local")  # Optional with default "local"

# Determine output directory
if OUTPUT_STORAGE == "drive":
    if DRIVE_DATA_DIR is None:
        raise ValueError("DRIVE_DATA_DIR not configured. Check .env file or use 'local' storage.")
    OUTPUT_DATA_DIR = DRIVE_DATA_DIR
    logger.info(f"Using Drive storage: {OUTPUT_DATA_DIR}")
elif OUTPUT_STORAGE == "local":
    OUTPUT_DATA_DIR = LOCAL_DATA_DIR
    logger.info(f"Using local storage: {OUTPUT_DATA_DIR}")
else:
    raise ValueError(f"Invalid output_storage: '{OUTPUT_STORAGE}'. Use 'local' or 'drive'.")

logger.info(f"Datasets to download: {DATASETS}")

# Create context
context = DownloadDataContext(
    output_data_dir=OUTPUT_DATA_DIR,
    datasets=DATASETS,
    output_storage=OUTPUT_STORAGE,
)

# Call main function
download_data(context)

logger.info("=" * 80)
logger.info("✓ download_data task completed successfully")
logger.info("=" * 80)

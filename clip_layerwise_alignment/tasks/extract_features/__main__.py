# tasks/extract_features/__main__.py

from pathlib import Path

from clip_layerwise_alignment.config import DRIVE_DATA_DIR, LOCAL_DATA_DIR
from clip_layerwise_alignment.utils import (
    check_missing_keys,
    load_config,
    setup_logging,
)

# Setup logging
script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

# Import task components
from clip_layerwise_alignment.tasks.extract_features import (
    ExtractFeaturesContext,
    extract_features,
)

logger.info("=" * 80)
logger.info("Starting extract_features task")
logger.info("=" * 80)

# Load config
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"
logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

# Validate config
required_keys = [
    "model_name",
    "pretrained",
    "text_layers",
    "image_layers",
    "extract_text_final",
    "extract_image_final",
    "input_dataset",
    "splits",
    "batch_size",
    "device",
]
check_missing_keys(required_keys, script_config)

# Parse config
MODEL_NAME = script_config["model_name"]
PRETRAINED = script_config["pretrained"]
TEXT_LAYERS = script_config["text_layers"]
IMAGE_LAYERS = script_config["image_layers"]
EXTRACT_TEXT_FINAL = script_config["extract_text_final"]
EXTRACT_IMAGE_FINAL = script_config["extract_image_final"]
INPUT_DATASET = script_config["input_dataset"]
SPLITS = script_config["splits"]
BATCH_SIZE = script_config["batch_size"]
DEVICE = script_config["device"]

# Optional config
NUM_WORKERS = script_config.get("num_workers", 4)  # Default to 4 workers
ENVIRONMENT = script_config.get("environment", "local")  # Default local

logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Model: {MODEL_NAME} ({PRETRAINED})")
logger.info(f"Dataset: {INPUT_DATASET}")
logger.info(f"Splits: {SPLITS}")
logger.info(f"Text layers: {TEXT_LAYERS}")
logger.info(f"Image layers: {IMAGE_LAYERS}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Batch size: {BATCH_SIZE}")

# Validate dataset
if INPUT_DATASET != "flickr30k":
    raise ValueError(f"Only 'flickr30k' dataset is currently supported, got '{INPUT_DATASET}'")

# Convert layer numbers to 0-indexed
text_layer_indices = [layer - 1 for layer in TEXT_LAYERS]
image_layer_indices = [layer - 1 for layer in IMAGE_LAYERS]

logger.info(f"Text layer indices (0-indexed): {text_layer_indices}")
logger.info(f"Image layer indices (0-indexed): {image_layer_indices}")

# Determine paths
if ENVIRONMENT == "local":
    data_root = LOCAL_DATA_DIR
elif ENVIRONMENT == "colab":
    if DRIVE_DATA_DIR is None:
        raise ValueError("DRIVE_DIR not configured. Mount Drive and set in .env")
    data_root = DRIVE_DATA_DIR
else:
    raise ValueError(f"Invalid environment: '{ENVIRONMENT}'")

# Same relative paths for both environments
flickr30k_dir = data_root / "raw" / INPUT_DATASET
output_dir = data_root / "processed" / INPUT_DATASET

# Validate flickr30k data exists
if ENVIRONMENT == "local":
    # Local: Check if unzipped dataset exists
    if not flickr30k_dir.exists():
        raise FileNotFoundError(
            f"Flickr30k data not found at {flickr30k_dir}. " "Please run download_data task first."
        )

    annotations_file = flickr30k_dir / "annotations" / "dataset_flickr30k.json"
    if not annotations_file.exists():
        raise FileNotFoundError(
            f"Flickr30k annotations not found at {annotations_file}. " "Dataset may be incomplete."
        )

elif ENVIRONMENT == "colab":
    # Colab: Check if zip file exists on Drive
    flickr30k_zip = flickr30k_dir.parent / "flickr30k.zip"

    if not flickr30k_zip.exists():
        raise FileNotFoundError(
            f"Flickr30k zip not found on Drive: {flickr30k_zip}\n"
            "Please run download_data task with output_storage='drive' first, "
            "or manually upload flickr30k.zip to Drive."
        )

    logger.info(f"✓ Found flickr30k.zip on Drive: {flickr30k_zip}")

logger.info(f"Input: {flickr30k_dir}")
logger.info(f"Output: {output_dir}")

# Create context
context = ExtractFeaturesContext(
    model_name=MODEL_NAME,
    pretrained=PRETRAINED,
    text_layers=TEXT_LAYERS,
    image_layers=IMAGE_LAYERS,
    text_layer_indices=text_layer_indices,
    image_layer_indices=image_layer_indices,
    extract_text_final=EXTRACT_TEXT_FINAL,
    extract_image_final=EXTRACT_IMAGE_FINAL,
    dataset=INPUT_DATASET,
    splits=SPLITS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    flickr30k_dir=flickr30k_dir,
    output_dir=output_dir,
    environment=ENVIRONMENT,
)

# Call main function
extract_features(context)

logger.info("=" * 80)
logger.info("✓ extract_features task completed successfully")
logger.info("=" * 80)

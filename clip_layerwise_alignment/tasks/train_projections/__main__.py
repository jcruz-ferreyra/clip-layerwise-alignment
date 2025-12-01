# tasks/train_projections/__main__.py

from pathlib import Path

from clip_layerwise_alignment.config import (
    DRIVE_DATA_DIR,
    DRIVE_MODELS_DIR,
    LOCAL_DATA_DIR,
    LOCAL_MODELS_DIR,
)
from clip_layerwise_alignment.utils import (
    check_missing_keys,
    load_config,
    setup_logging,
)

# Setup logging
script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

# Import task components
from clip_layerwise_alignment.tasks.train_projections import (
    TrainProjectionsContext,
    train_projections,
)

logger.info("=" * 80)
logger.info("Starting train_projections task")
logger.info("=" * 80)

# Load config
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"
logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

# Validate config
required_keys = [
    "text_layer",
    "image_layer",
    "projection_type",
    "dataset",
    "checkpoints_subdir",
]
check_missing_keys(required_keys, script_config)

# Parse config
TEXT_LAYER = script_config["text_layer"]
IMAGE_LAYER = script_config["image_layer"]
PROJECTION_TYPE = script_config["projection_type"]
DATASET = script_config["dataset"]
CHECKPOINTS_SUBDIR = script_config["checkpoints_subdir"]

# Optional configurations with default fallback
TRAINING_PARAMS = script_config.get("training_params", {})  # filled in context creation
USE_LAYERNORM = script_config.get("use_layernorm", False)
DEVICE = script_config.get("device", None)
NUM_WORKERS = script_config.get("num_workers", 4)
ENVIRONMENT = script_config.get("environment", "local")

# Auto-detect device if not specified
if DEVICE is None:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Auto-detected device: {DEVICE}")
else:
    logger.info(f"Using configured device: {DEVICE}")

logger.info(f"Layer pair: text {TEXT_LAYER} - image {IMAGE_LAYER}")
logger.info(f"Dataset: {DATASET}")
logger.info(f"Environment: {ENVIRONMENT}")

# Validate dataset
if DATASET != "flickr30k":
    raise ValueError(f"Only 'flickr30k' dataset supported, got '{DATASET}'")

# Environment-based directory selection
if ENVIRONMENT == "colab":
    if DRIVE_DATA_DIR is None or DRIVE_MODELS_DIR is None:
        raise ValueError("DRIVE_DIR not configured. Mount Drive and set in .env")
    data_root = DRIVE_DATA_DIR
    model_root = DRIVE_MODELS_DIR
    logger.info("Using Drive storage (Colab)")
elif ENVIRONMENT == "local":
    data_root = LOCAL_DATA_DIR
    model_root = LOCAL_MODELS_DIR
    logger.info("Using local storage")
else:
    raise ValueError(f"Invalid environment: '{ENVIRONMENT}'")

# Construct paths
features_dir = data_root / "processed" / DATASET
checkpoints_dir = model_root / CHECKPOINTS_SUBDIR

logger.info(f"Features: {features_dir}")
logger.info(f"Checkpoints: {checkpoints_dir}")

# Validate features directory exists
if not features_dir.exists():
    raise FileNotFoundError(
        f"Features directory not found: {features_dir}\n" "Please run extract_features task first."
    )

# Create context
context = TrainProjectionsContext(
    text_layer=TEXT_LAYER,
    image_layer=IMAGE_LAYER,
    training_params=TRAINING_PARAMS,
    projection_type=PROJECTION_TYPE,
    use_layernorm=USE_LAYERNORM,
    dataset=DATASET,
    device=DEVICE,
    num_workers=NUM_WORKERS,
    features_dir=features_dir,
    checkpoints_dir=checkpoints_dir,
    environment=ENVIRONMENT,
)

# Call main function
train_projections(context)

logger.info("=" * 80)
logger.info("✓ train_projections task completed successfully")
logger.info("=" * 80)

# tasks/project_eval_samples/__main__.py

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
from clip_layerwise_alignment.tasks.project_eval_samples import (
    ProjectEvalContext,
    project_eval_samples,
)

logger.info("=" * 80)
logger.info("Starting project_eval_samples task")
logger.info("=" * 80)

# Load config
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"
logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

# Validate config
required_keys = [
    "layer_pairs",
    "projection_type",
    "dataset",
    "eval_split",
    "projections_subdir",
]
check_missing_keys(required_keys, script_config)

# Parse config
LAYER_PAIRS = script_config["layer_pairs"]
PROJECTION_TYPE = script_config["projection_type"]
USE_LAYERNORM = script_config.get("use_layernorm", False)
DATASET = script_config["dataset"]
EVAL_SPLIT = script_config["eval_split"]
PROJECTIONS_SUBDIR = script_config["projections_subdir"]
ENVIRONMENT = script_config.get("environment", "local")

logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Dataset: {DATASET}")
logger.info(f"Eval split: {EVAL_SPLIT}")
logger.info(f"Layer pairs to project: {len(LAYER_PAIRS)}")

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
projections_dir = model_root / PROJECTIONS_SUBDIR
output_dir = data_root / "results" / DATASET

logger.info(f"Features: {features_dir}")
logger.info(f"Projections: {projections_dir}")
logger.info(f"Output: {output_dir}")

# Validate directories exist
eval_features_dir = features_dir / EVAL_SPLIT
if not eval_features_dir.exists():
    raise FileNotFoundError(
        f"Eval features not found: {eval_features_dir}\n" "Please run extract_features task first."
    )

if not projections_dir.exists():
    raise FileNotFoundError(
        f"Projections directory not found: {projections_dir}\n"
        "Please run train_projections task first."
    )

# Create context
context = ProjectEvalContext(
    layer_pairs=LAYER_PAIRS,
    dataset=DATASET,
    eval_split=EVAL_SPLIT,
    features_dir=features_dir,
    projections_dir=projections_dir,
    output_dir=output_dir,
    projection_type=PROJECTION_TYPE,
    use_layernorm=USE_LAYERNORM,
    environment=ENVIRONMENT,
)

# Call main function
project_eval_samples(context)

logger.info("=" * 80)
logger.info("✓ project_eval_samples task completed successfully")
logger.info("=" * 80)

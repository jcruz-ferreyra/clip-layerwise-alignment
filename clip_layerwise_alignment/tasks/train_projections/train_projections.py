# tasks/train_projections/train_projections.py

import copy
import logging
from typing import Any, Dict, List, Tuple

from clip_layerwise_alignment.utils import create_projection_model
import torch
import torch.nn as nn

from .dataset import create_contrastive_dataloader
from .early_stopping import _create_early_stopping
from .optimizer import _create_optimizer
from .scheduler import _create_scheduler
from .training import _run_training_loop, _run_val_loop
from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def _load_features_for_pair(ctx: TrainProjectionsContext) -> Dict[str, Dict[str, Any]]:
    """
    Load pre-extracted features for training and validation.

    Returns:
        Dictionary with 'train' and optionally 'val' keys:
        {
            'train': {
                'text_features': [N, dim],
                'image_features': [N, dim],
                'metadata': {...}
            },
            'val': {  # Only if val split exists
                'text_features': [N, dim],
                'image_features': [N, dim],
                'metadata': {...}
            }
        }
    """
    logger.info("Loading features for layer pair...")

    # Determine filenames
    text_filename = (
        f"text_{ctx.text_layer}.pt"
        if ctx.text_layer == "final"
        else f"text_layer_{ctx.text_layer}.pt"
    )
    image_filename = (
        f"image_{ctx.image_layer}.pt"
        if ctx.image_layer == "final"
        else f"image_layer_{ctx.image_layer}.pt"
    )

    features = {}

    # === Load train features ===
    text_file_train = ctx.features_dir_train / text_filename
    image_file_train = ctx.features_dir_train / image_filename

    if not text_file_train.exists():
        raise FileNotFoundError(f"Train text features not found: {text_file_train}")
    if not image_file_train.exists():
        raise FileNotFoundError(f"Train image features not found: {image_file_train}")

    logger.info(f"Loading train from {ctx.features_dir_train.name}/")
    train_text_data = torch.load(text_file_train)
    train_image_data = torch.load(image_file_train)

    features["train"] = {
        "text_features": train_text_data["features"].detach(),
        "image_features": train_image_data["features"].detach(),
        "metadata": train_text_data["metadata"],
    }

    # Validate
    if len(features["train"]["text_features"]) != len(features["train"]["image_features"]):
        raise ValueError("Train feature count mismatch")

    logger.info(f"  Train: {len(features['train']['text_features'])} samples")
    logger.info(f"    Text: {features['train']['text_features'].shape}")
    logger.info(f"    Image: {features['train']['image_features'].shape}")

    # === Load val features (if exists) ===
    if ctx.features_dir_val is not None:
        text_file_val = ctx.features_dir_val / text_filename
        image_file_val = ctx.features_dir_val / image_filename

        logger.info(f"Loading val from {ctx.features_dir_val.name}/")
        val_text_data = torch.load(text_file_val)
        val_image_data = torch.load(image_file_val)

        features["val"] = {
            "text_features": val_text_data["features"].detach(),
            "image_features": val_image_data["features"].detach(),
            "metadata": val_text_data["metadata"],
        }

        logger.info(f"  Val: {len(features['val']['text_features'])} samples")
    else:
        logger.info("  No separate val split")

    logger.info("✓ Features loaded successfully")

    return features


def _create_projection_model(ctx: TrainProjectionsContext):
    """Create the projection model using context parameters"""
    projection_model = create_projection_model(
        text_layer=ctx.text_layer,
        image_layer=ctx.image_layer,
        projection_type=ctx.projection_type,
        use_layernorm=ctx.use_layernorm,
        learnable_temperature=ctx.training_params["learnable_temperature"],
        initial_temperature=ctx.training_params["temperature"],
        device=ctx.device,
    )

    return projection_model


def _train_projection(
    ctx: TrainProjectionsContext, model: nn.Module, features: Dict[str, Dict]
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Train projection model via contrastive learning.

    Returns:
        Tuple of (trained_model, history)
        where history is a list of dicts with keys: epoch, train_loss, val_loss, lr
    """

    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)

    # Create dataloaders
    train_dataloader = create_contrastive_dataloader(ctx, features["train"], split="train")

    val_dataloader = None
    if "val" in features:
        val_dataloader = create_contrastive_dataloader(ctx, features["val"], split="val")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    else:
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        logger.info("No validation split")

    logger.info(f"Batches per epoch: {len(train_dataloader)}")

    # Create optimizer
    optimizer = _create_optimizer(ctx, model)

    # Create scheduler
    scheduler = _create_scheduler(ctx, optimizer)

    # Create early stopping (if validation exists)
    early_stopping = None
    best_model_state = None
    best_val_loss = float("inf")
    if val_dataloader is not None:
        early_stopping = _create_early_stopping(ctx)

    # Training history
    history = []

    # Training loop
    for epoch in range(ctx.training_params["epochs"]):
        # Train epoch
        train_loss = _run_training_loop(ctx, model, train_dataloader, optimizer, epoch)
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        # Step scheduler (once per epoch)
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"           Learning Rate = {current_lr:.2e}")

        # Initialize epoch record
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": None,  # Will be filled if validation exists
            "lr": current_lr,
        }

        # Validation epoch
        if val_dataloader is not None:
            val_loss = _run_val_loop(ctx, model, val_dataloader, epoch)
            logger.info(f"           Val Loss = {val_loss:.4f}")

            epoch_record["val_loss"] = val_loss

            # Update best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                logger.info("           ✓ New best model")

            # Check early stopping
            if early_stopping(val_loss, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                history.append(epoch_record)  # Save last epoch before breaking
                break

        # Add epoch to history
        history.append(epoch_record)

    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

    logger.info("=" * 80)
    logger.info("✓ Training complete")
    logger.info("=" * 80)

    return model, history


def _save_projection(model: nn.Module, ctx: TrainProjectionsContext) -> None:
    """
    Save trained projection model and training metadata.

    Args:
        model: Trained ProjectionWrapper model
        ctx: Training configuration context
    """
    logger.info("Saving trained projection...")

    # Prepare checkpoint data
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "text_layer": ctx.text_layer,
            "image_layer": ctx.image_layer,
            "projection_type": ctx.projection_type,
            "use_layernorm": ctx.use_layernorm,
            "training_params": ctx.training_params,
        },
        "projection_name": ctx.projection_name,
    }

    # Save to disk
    checkpoint_path = ctx.checkpoint_path
    torch.save(checkpoint, checkpoint_path)

    # Log file info
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Checkpoint saved: {checkpoint_path.name}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Location: {checkpoint_path.parent}")


def _save_training_history(ctx: TrainProjectionsContext, history: List[Dict[str, float]]):
    """Save training history to JSON."""
    history_path = ctx.checkpoints_dir / f"{ctx.projection_name}_history.json"

    import json

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"✓ Training history saved: {history_path.name}")


# ============================================================================
# Main public function
# ============================================================================


def train_projections(ctx: TrainProjectionsContext) -> None:
    """
    Train a single projection layer pair via contrastive learning.

    Main orchestrator that:
    1. Loads pre-extracted features for specified layer pair
    2. Creates projection model (linear + optional LayerNorm)
    3. Trains via contrastive loss
    4. Saves trained weights

    No colab unzipping needed - loads .pt files directly from Drive/local.

    Args:
        ctx: TrainProjectionsContext with layer pair, hyperparameters, and paths
    """
    logger.info("=" * 80)
    logger.info("Starting projection training")
    logger.info("=" * 80)

    logger.info("Layer pair:")
    logger.info(f"  Text layer: {ctx.text_layer}")
    logger.info(f"  Image layer: {ctx.image_layer}")

    logger.info("Training config:")
    logger.info(f"  Learning rate: {ctx.training_params['learning_rate']}")
    logger.info(f"  Batch size: {ctx.training_params['batch_size']}")
    logger.info(f"  Epochs: {ctx.training_params['epochs']}")
    logger.info(f"  Temperature: {ctx.training_params['temperature']}")
    logger.info(f"  Use LayerNorm: {ctx.use_layernorm}")
    logger.info(f"  Device: {ctx.device}")

    # Load pre-extracted features for this pair
    features = _load_features_for_pair(ctx)

    # Create projection model
    projection_model = _create_projection_model(ctx)

    # Train the projection (handles dataset/dataloader internally)
    trained_model, history = _train_projection(ctx, projection_model, features)

    # Save trained weights
    _save_projection(trained_model, ctx)

    # Save history
    _save_training_history(ctx, history)

    logger.info("=" * 80)
    logger.info("✓ Projection training complete!")
    logger.info("=" * 80)

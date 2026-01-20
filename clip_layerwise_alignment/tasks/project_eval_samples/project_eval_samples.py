# tasks/project_eval_samples/project_eval_samples.py

import logging
from typing import Dict

import torch
import torch.nn as nn

from clip_layerwise_alignment.utils import create_projection_model

from .types import ProjectEvalContext

logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================

def _load_eval_features_for_pair(pair: Dict, ctx: ProjectEvalContext) -> Dict[str, torch.Tensor]:
    """
    Load evaluation features for the specified layer pair.

    Only loads features that need projection (the intermediate layer).
    Final features don't need projection as they're already in shared space.

    Args:
        pair: Dict with 'text_layer' and 'image_layer'
        ctx: Evaluation context

    Returns:
        Dict with loaded features:
        - If text intermediate: {'text_features': [N, dim], 'metadata': {...}}
        - If image intermediate: {'image_features': [N, dim], 'metadata': {...}}
    """
    text_layer = pair["text_layer"]
    image_layer = pair["image_layer"]

    text_is_final = text_layer == "final"
    image_is_final = image_layer == "final"

    # Both intermediate not supported (haven't trained these)
    if not text_is_final and not image_is_final:
        raise NotImplementedError(
            f"Both layers are intermediate (text_L{text_layer}, image_L{image_layer}). "
            "Only implemented projections from intermediate → final."
        )

    # Both final makes no sense (nothing to project)
    if text_is_final and image_is_final:
        raise ValueError("Both layers are final - no projection needed!")

    # Load text features (if text is intermediate)
    if not text_is_final:
        text_filename = f"text_layer_{text_layer}.pt"
        text_file = ctx.eval_features_dir / text_filename

        if not text_file.exists():
            raise FileNotFoundError(f"Text features not found: {text_file}")

        logger.info(f"Loading text features: {text_filename}")
        text_data = torch.load(text_file)

        return {"text_features": text_data["features"].detach(), "metadata": text_data["metadata"]}

    # Load image features (if image is intermediate)
    if not image_is_final:
        image_filename = f"image_layer_{image_layer}.pt"
        image_file = ctx.eval_features_dir / image_filename

        if not image_file.exists():
            raise FileNotFoundError(f"Image features not found: {image_file}")

        logger.info(f"Loading image features: {image_filename}")
        image_data = torch.load(image_file)

        return {
            "image_features": image_data["features"].detach(),
            "metadata": image_data["metadata"],
        }


def _load_trained_projection(pair: Dict, ctx: ProjectEvalContext) -> nn.Module:
    """
    Load trained projection model from checkpoint.

    Args:
        pair: Dict with 'text_layer' and 'image_layer'
        ctx: Evaluation context

    Returns:
        Loaded model in eval mode
    """
    text_layer = pair["text_layer"]
    image_layer = pair["image_layer"]

    # Get checkpoint path
    checkpoint_path = ctx.get_projection_path(text_layer, image_layer)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Projection checkpoint not found: {checkpoint_path}\n"
            f"Please train projection for text_L{text_layer} → image_L{image_layer} first."
        )

    logger.info(f"Loading projection: {checkpoint_path.name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    config = checkpoint["config"]
    logger.info(f"  Projection type: {config['projection_type']}")
    logger.info(f"  Use LayerNorm: {config['use_layernorm']}")

    # Create projection model
    projection_model = create_projection_model(
        text_layer=text_layer,
        image_layer=image_layer,
        projection_type=ctx.projection_type,
        use_layernorm=ctx.use_layernorm,
    )

    # Load trained weights
    projection_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Set to evaluation mode
    projection_model.eval()

    logger.info("✓ Projection loaded successfully")

    return projection_model


def _project_features(
    model: nn.Module, eval_features: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Project features through trained model.

    Args:
        model: Trained ProjectionWrapper
        eval_features: Dict with features to project

    Returns:
        Dict with both projected embeddings
    """
    logger.info("Projecting features...")

    with torch.no_grad():
        if "text_features" in eval_features:
            # Text intermediate, need dummy image
            text_feats = eval_features["text_features"]
            dummy_image = torch.zeros_like(text_feats)  # Same shape, dummy data

            text_embed, _ = model(text_feats, dummy_image)

            return {"text_embeddings": text_embed, "metadata": eval_features["metadata"]}
        else:
            # Image intermediate, need dummy text
            image_feats = eval_features["image_features"]
            dummy_text = torch.zeros(len(image_feats), 512)  # [N, 512] dummy

            _, image_embed = model(dummy_text, image_feats)

            return {"image_embeddings": image_embed, "metadata": eval_features["metadata"]}


def _save_projected_embeddings(
    projected_embeddings: Dict[str, torch.Tensor], pair: Dict, ctx: ProjectEvalContext
) -> None:
    """
    Save projected embeddings to disk.

    Args:
        projected_embeddings: Dict with embeddings and metadata
        pair: Layer pair configuration
        ctx: Evaluation context
    """
    logger.info("Saving projected embeddings...")

    # Prepare save data
    save_data = {
        "layer_pair": {
            "text_layer": pair["text_layer"],
            "image_layer": pair["image_layer"],
        },
        "eval_split": ctx.eval_split,
    }

    # Add embeddings (text or image)
    if "text_embeddings" in projected_embeddings:
        save_data["text_embeddings"] = projected_embeddings["text_embeddings"]
        save_data["modality"] = "text"
    else:
        save_data["image_embeddings"] = projected_embeddings["image_embeddings"]
        save_data["modality"] = "image"

    # Add metadata
    save_data["metadata"] = projected_embeddings["metadata"]

    # Save to output path
    output_path = ctx.get_output_path(pair["text_layer"], pair["image_layer"])
    torch.save(save_data, output_path)

    # Log
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved: {output_path.name}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")


# ============================================================================
# Main public function
# ============================================================================

def project_eval_samples(ctx: ProjectEvalContext) -> None:
    """
    Project evaluation samples through trained projection layers.

    Main orchestrator that:
    1. For each layer pair:
       - Loads evaluation features
       - Loads trained projection
       - Projects features to shared space
       - Saves projected embeddings

    Args:
        ctx: ProjectEvalContext with layer pairs, paths, and configuration
    """
    logger.info("=" * 80)
    logger.info("Projecting evaluation samples")
    logger.info("=" * 80)
    logger.info(f"Dataset: {ctx.dataset}")
    logger.info(f"Eval split: {ctx.eval_split}")
    logger.info(f"Number of layer pairs: {len(ctx.layer_pairs)}")

    # Process each layer pair
    for idx, pair in enumerate(ctx.layer_pairs):
        logger.info("=" * 80)
        logger.info(f"[{idx+1}/{len(ctx.layer_pairs)}] Processing pair:")
        logger.info(f"  Text layer: {pair['text_layer']}")
        logger.info(f"  Image layer: {pair['image_layer']}")
        logger.info("=" * 80)

        # Load evaluation features for this pair
        eval_features = _load_eval_features_for_pair(pair, ctx)

        # Load trained projection
        projection_model = _load_trained_projection(pair, ctx)

        # Project features through trained model
        projected_embeddings = _project_features(projection_model, eval_features)

        # Save projected embeddings
        _save_projected_embeddings(projected_embeddings, pair, ctx)

        logger.info(f"✓ Pair {idx+1}/{len(ctx.layer_pairs)} complete")

    logger.info("=" * 80)
    logger.info("✓ All evaluation samples projected!")
    logger.info("=" * 80)

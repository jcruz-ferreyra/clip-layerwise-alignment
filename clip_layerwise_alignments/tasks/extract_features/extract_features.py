# tasks/extract_features/extract_features.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_layerwise_alignments.colab import retrieve_and_unzip_data

from .flickr30k import (
    _extract_flickr30k_img_features,
    _extract_flickr30k_txt_features,
    _load_flickr30k_dataset,
)
from .flickr30k import _save_flickr30k_features  # ← ADD
from .types import ExtractFeaturesContext

logger = logging.getLogger(__name__)


def _load_clip_model(ctx: ExtractFeaturesContext) -> Tuple[nn.Module, Any, Any]:
    """Load pretrained CLIP model and preprocessing functions."""
    import open_clip

    logger.info(f"Loading CLIP model: {ctx.model_name}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        ctx.model_name, pretrained=ctx.pretrained
    )

    tokenizer = open_clip.get_tokenizer(ctx.model_name)

    model = model.to(ctx.device)
    model.eval()  # Inference mode

    logger.info(f"✓ Model loaded on {ctx.device}")

    return model, preprocess, tokenizer


def _extract_features_for_split(
    model: nn.Module, preprocess: Any, tokenizer: Any, ctx: ExtractFeaturesContext, split: str
):
    """Extract features for one data split (train/test)."""
    logger.info("=" * 80)
    logger.info(f"Processing {split} split")
    logger.info("=" * 80)

    if ctx.dataset == "flickr30k":
        # Load dataset annotations
        split_dataset = _load_flickr30k_dataset(ctx, split)

        # Extract image features
        img_features = _extract_flickr30k_img_features(model, preprocess, ctx, split_dataset)

        # Extract text features
        txt_features = _extract_flickr30k_txt_features(model, tokenizer, ctx, split_dataset)

        # Save features
        _save_flickr30k_features(img_features, txt_features, ctx, split)

        logger.info(f"✓ {split} split complete")


def extract_features(ctx: ExtractFeaturesContext) -> None:
    """
    Extract CLIP features at multiple layers for both modalities.

    Main orchestrator function that:
    1. Loads pretrained CLIP model
    2. For each split (train/test):
       - Loads Flickr30k data
       - Extracts vision features at specified layers
       - Extracts text features at specified layers
       - Saves each layer to separate file

    Args:
        ctx: ExtractFeaturesContext with model config, layer indices, and paths
    """
    logger.info("=" * 80)
    logger.info("Starting feature extraction process")
    logger.info("=" * 80)
    logger.info(f"Model: {ctx.model_name}")
    logger.info(f"Text layers: {ctx.text_layers}")
    logger.info(f"Image layers: {ctx.image_layers}")
    logger.info(f"Splits: {ctx.splits}")

    # Colab: Extract dataset from Drive to local SSD
    if ctx.environment == "colab":
        source_zip = ctx.flickr30k_dir.parent / "flickr30k.zip"
        colab_flickr30k_dir = retrieve_and_unzip_data(
            source_zip=source_zip, extract_to=Path("/content/data/raw"), dataset_name="flickr30k"
        )
        # Update context to use Colab local path (fast!)
        ctx.flickr30k_dir = colab_flickr30k_dir
        logger.info(f"Updated input path to: {ctx.flickr30k_dir}")

    # Load CLIP model
    model, preprocess, tokenizer = _load_clip_model(ctx)

    # Extract features for each split
    for split in ctx.splits:
        _extract_features_for_split(model, preprocess, tokenizer, ctx, split)

    logger.info("=" * 80)
    logger.info("✓ Feature extraction complete!")
    logger.info("=" * 80)

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import open_clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .clip import _extract_text_features, _extract_vision_features
from .types import ExtractFeaturesContext

logger = logging.getLogger(__name__)


class Flickr30kImageDataset(Dataset):
    def __init__(self, image_data_list, images_dir, preprocess):
        self.images = image_data_list  # List of image dicts
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_data = self.images[idx]

        img_path = self.images_dir / img_data["filename"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        return {
            "image": image_tensor,
            "filename": img_data["filename"],
            "imgid": img_data["imgid"],
        }


class Flickr30kTextDataset(Dataset):
    """Extract features from all captions (5 per image)."""

    def __init__(self, image_data_list, tokenizer):
        # Flatten: create one entry per caption
        self.caption_data = []

        for img_data in image_data_list:
            for sent in img_data["sentences"]:
                self.caption_data.append(
                    {
                        "text": sent["raw"],
                        "filename": img_data["filename"],
                        "imgid": img_data["imgid"],
                        "sentid": sent["sentid"],
                    }
                )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        data = self.caption_data[idx]

        # Tokenize (returns [77] tensor)
        text_tokens = self.tokenizer([data["text"]])[0]

        return {
            "text": text_tokens,
            "caption": data["text"],
            "filename": data["filename"],
            "imgid": data["imgid"],
        }


def _load_flickr30k_dataset(ctx: ExtractFeaturesContext, split: str):
    """
    Load Flickr30k dataset for given split.

    Returns dataset/dataloader that yields (images, texts, metadata) batches.
    """
    logger.info(f"Loading Flickr30k {split} split...")

    # Load annotations JSON
    annotations_path = ctx.flickr30k_dir / "annotations" / "dataset_flickr30k.json"
    with open(annotations_path) as f:
        data = json.load(f)

    # Filter by split
    split_dataset = [img for img in data["images"] if img["split"] == split]

    logger.info(f"Found {len(split_dataset)} images in {split} split")
    logger.info(f"Total captions: {len(split_dataset) * 5}")

    return split_dataset


def _extract_flickr30k_img_features(
    model: nn.Module, preprocess: Any, ctx: ExtractFeaturesContext, split_dataset: List[Dict]
) -> Dict[str, Any]:
    """
    Extract image features for all images in split.

    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        ctx: Configuration context
        split_dataset: List of image dicts from JSON annotations

    Returns:
        Dictionary with layer features and metadata:
        {
            'layer_1': torch.Tensor([N, 768]),
            'layer_3': torch.Tensor([N, 768]),
            ...
            'final': torch.Tensor([N, 512]),
            'metadata': {
                'filenames': List[str],
                'imgids': List[int]
            }
        }
    """
    logger.info(f"Extracting image features from {len(split_dataset)} images")

    # Step 1: Create dataset
    dataset = Flickr30kImageDataset(split_dataset, ctx.flickr30k_dir / "images", preprocess)

    # Step 2: Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=ctx.batch_size,
        shuffle=False,  # Keep order for easier debugging/matching
        num_workers=ctx.num_workers,
        pin_memory=True if ctx.device == "cuda" else False,
    )

    # Step 3: Initialize accumulators
    all_features = {}
    metadata = {"filenames": [], "imgids": []}

    # Step 4: Process batches
    for batch in tqdm(loader, desc="Extracting image features"):
        images = batch["image"].to(ctx.device)  # [B, 3, 224, 224]

        # Extract features for this batch
        batch_intermediates, batch_final = _extract_vision_features(
            model, images, ctx.image_layer_indices
        )

        # Accumulate intermediate features
        for layer_name, features in batch_intermediates.items():
            if layer_name not in all_features:
                all_features[layer_name] = []
            all_features[layer_name].append(features.cpu())

        # Accumulate final features (if requested)
        if ctx.extract_image_final:
            if "final" not in all_features:
                all_features["final"] = []
            all_features["final"].append(batch_final.cpu())

        # Accumulate metadata
        metadata["filenames"].extend(batch["filename"])
        metadata["imgids"].extend(
            batch["imgid"].tolist() if torch.is_tensor(batch["imgid"]) else batch["imgid"]
        )

    # Step 5: Concatenate batches into single tensors
    logger.info("Concatenating image features across batches...")
    for layer_name in all_features:
        all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)
        logger.info(f"  {layer_name}: {all_features[layer_name].shape}")

    logger.info(f"✓ Extracted features for {len(metadata['filenames'])} images")

    # Step 6: Return organized structure
    return {**all_features, "metadata": metadata}


def _extract_flickr30k_txt_features(
    model: nn.Module, tokenizer: Any, ctx: ExtractFeaturesContext, split_dataset: List[Dict]
) -> Dict[str, Any]:
    """
    Extract text features for all captions in split.

    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        ctx: Configuration context
        split_dataset: List of image dicts from JSON annotations

    Returns:
        Dictionary with layer features and metadata:
        {
            'layer_1': torch.Tensor([N*5, 512]),
            'layer_3': torch.Tensor([N*5, 512]),
            ...
            'final': torch.Tensor([N*5, 512]),
            'metadata': {
                'captions': List[str],
                'filenames': List[str],
                'imgids': List[int]
            }
        }
    """
    logger.info(f"Extracting text features from {len(split_dataset) * 5} captions")

    # Step 1: Create dataset
    dataset = Flickr30kTextDataset(split_dataset, tokenizer)

    # Step 2: Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=ctx.batch_size,
        shuffle=False,  # Keep order for pairing with images
        num_workers=ctx.num_workers,
        pin_memory=True if ctx.device == "cuda" else False,
    )

    # Step 3: Initialize accumulators
    all_features = {}
    metadata = {"captions": [], "filenames": [], "imgids": []}

    # Step 4: Process batches
    for batch in tqdm(loader, desc="Extracting text features"):
        texts = batch["text"].to(ctx.device)  # [B, 77]

        # Extract features for this batch
        batch_intermediates, batch_final = _extract_text_features(
            model, texts, ctx.text_layer_indices
        )

        # Accumulate intermediate features
        for layer_name, features in batch_intermediates.items():
            if layer_name not in all_features:
                all_features[layer_name] = []
            all_features[layer_name].append(features.cpu())

        # Accumulate final features (if requested)
        if ctx.extract_text_final:
            if "final" not in all_features:
                all_features["final"] = []
            all_features["final"].append(batch_final.cpu())

        # Accumulate metadata
        metadata["captions"].extend(batch["caption"])
        metadata["filenames"].extend(batch["filename"])
        metadata["imgids"].extend(
            batch["imgid"].tolist() if torch.is_tensor(batch["imgid"]) else batch["imgid"]
        )

    # Step 5: Concatenate batches into single tensors
    logger.info("Concatenating text features across batches...")
    for layer_name in all_features:
        all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)
        logger.info(f"  {layer_name}: {all_features[layer_name].shape}")

    logger.info(f"✓ Extracted features for {len(metadata['captions'])} captions")

    # Step 6: Return organized structure
    return {**all_features, "metadata": metadata}


def _validate_caption_counts(img_metadata, txt_metadata):
    """Validate and report caption distribution."""
    from collections import Counter

    # Count captions per image
    caption_counts = Counter(txt_metadata["imgids"])

    # Report statistics
    counts_distribution = Counter(caption_counts.values())
    logger.info("Caption count distribution:")
    for count, num_images in sorted(counts_distribution.items()):
        logger.info(f"  {num_images} images with {count} captions")

    # Check for expected pattern (should be 5 for Flickr30k)
    if len(counts_distribution) > 1:
        logger.warning("Variable caption counts detected!")
        # Show examples
        for imgid, count in caption_counts.most_common(5):
            if count != 5:
                logger.warning(f"  Image {imgid} has {count} captions (expected 5)")


def _repeat_image_features_robust(img_features, img_metadata, txt_metadata):
    """
    Repeat image features to match text captions using imgid matching.

    Handles variable caption counts per image.
    """
    # Build fast lookup: imgid -> image feature index
    imgid_to_idx = {imgid: idx for idx, imgid in enumerate(img_metadata["imgids"])}

    repeated = []
    missing_count = 0

    for txt_imgid in txt_metadata["imgids"]:
        if txt_imgid in imgid_to_idx:
            img_idx = imgid_to_idx[txt_imgid]
            repeated.append(img_features[img_idx])
        else:
            logger.warning(f"Caption references missing image: imgid={txt_imgid}")
            missing_count += 1
            # Could skip or use zero vector

    if missing_count > 0:
        logger.error(f"Found {missing_count} captions with missing images!")

    return torch.stack(repeated)


def _save_flickr30k_features(img_features, txt_features, ctx, split):
    """Save features with validation and robust pairing."""

    # Step 1: Validate caption counts
    _validate_caption_counts(img_features["metadata"], txt_features["metadata"])

    # Step 2: Create robust image-text pairing
    paired_img_features = {}
    for layer_name, features in img_features.items():
        if layer_name == "metadata":
            continue

        paired_img_features[layer_name] = _repeat_image_features_robust(
            features, img_features["metadata"], txt_features["metadata"]
        )

    # Step 3: Save each layer separately
    output_dir = ctx.output_dir / f"{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image features (repeated to match captions)
    for layer_name, features in paired_img_features.items():
        torch.save(
            {
                "features": features,
                "metadata": txt_features["metadata"],  # Use text metadata (145k)
                "layer": layer_name,
            },
            output_dir / f"image_{layer_name}.pt",
        )

    # Save text features
    for layer_name, features in txt_features.items():
        if layer_name == "metadata":
            continue
        torch.save(
            {
                "features": features,
                "metadata": txt_features["metadata"],
                "layer": layer_name,
            },
            output_dir / f"text_{layer_name}.pt",
        )

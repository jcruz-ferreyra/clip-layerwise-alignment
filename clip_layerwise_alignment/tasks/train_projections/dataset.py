# tasks/train_projections/dataset.py

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ContrastiveFeaturesDataset(Dataset):
    """Dataset with on-the-fly random caption selection."""

    def __init__(self, text_feats, image_feats, imgids):
        self.text_features = text_feats
        self.image_features = image_feats
        self.imgids = np.array(imgids)

        # Build mapping
        self.img_to_captions = {}
        for idx, imgid in enumerate(imgids):
            if imgid not in self.img_to_captions:
                self.img_to_captions[imgid] = []
            self.img_to_captions[imgid].append(idx)

        self.unique_imgids = sorted(self.img_to_captions.keys())

    def __len__(self):
        return len(self.unique_imgids)  # 29,000 instead of 145,000

    def __getitem__(self, idx):
        # idx is between 0 and 28,999
        imgid = self.unique_imgids[idx]

        # Randomly pick one of the 5 captions for this image
        caption_indices = self.img_to_captions[imgid]
        caption_idx = np.random.choice(caption_indices)

        return {"text": self.text_features[caption_idx], "image": self.image_features[caption_idx]}


def create_contrastive_dataloader(ctx, features_dict, split="train"):
    """
    Create dataloader for contrastive learning.

    Args:
        ctx: Training context
        features_dict: Dict with 'text_features' and 'image_features'
        split: "train" or "val"

    Returns:
        DataLoader configured for the split
    """
    # Create dataset
    dataset = ContrastiveFeaturesDataset(
        text_feats=features_dict["text_features"],
        image_feats=features_dict["image_features"],
        imgids=features_dict["metadata"]["imgids"],
    )

    # Configure based on split
    if split == "train":
        dataloader = DataLoader(
            dataset,
            batch_size=ctx.training_params["batch_size"],
            shuffle=True,  # Shuffle for training
            num_workers=ctx.num_workers,
            pin_memory=True if ctx.device == "cuda" else False,
            drop_last=True,  # Stable batch size for contrastive loss
        )
    elif split == "val":
        dataloader = DataLoader(
            dataset,
            batch_size=ctx.training_params["batch_size"],
            shuffle=False,  # No shuffle for validation
            num_workers=ctx.num_workers,
            pin_memory=True if ctx.device == "cuda" else False,
            drop_last=False,  # Evaluate on all samples
        )
    else:
        raise ValueError(f"Invalid split: {split}")

    return dataloader

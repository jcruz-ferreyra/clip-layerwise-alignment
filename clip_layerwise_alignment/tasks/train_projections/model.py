# tasks/train_projections/train_projections.py

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


class ProjectionWrapper(nn.Module):
    """Wrapper that handles projection, normalization, and temperature."""

    def __init__(
        self, text_projection, image_projection, learnable_temperature, initial_temperature
    ):
        super().__init__()

        self.text_projection = text_projection
        self.image_projection = image_projection

        # Temperature parameter
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.ones([]) * np.log(1 / initial_temperature))
        else:
            # Constant temperature: buffer makes it a tensor (movable to cuda, saveable as pt)
            self.register_buffer("temperature", torch.tensor(initial_temperature))

    def forward(self, text_features, image_features):
        """
        Project features to shared space and normalize.

        Returns:
            text_embed: [B, 512] normalized embeddings
            image_embed: [B, 512] normalized embeddings
        """
        # Project (or pass through if None)
        if self.text_projection is not None:
            text_proj = self.text_projection(text_features)
        else:
            text_proj = text_features  # Already in shared space

        if self.image_projection is not None:
            image_proj = self.image_projection(image_features)
        else:
            image_proj = image_features  # Already in shared space

        # Normalize to unit sphere
        text_embed = F.normalize(text_proj, dim=-1)
        image_embed = F.normalize(image_proj, dim=-1)

        return text_embed, image_embed

    @property
    def temp(self):
        """Get current temperature value."""
        if hasattr(self, "log_temperature"):
            return self.log_temperature.exp()
        else:
            return self.temperature


def _create_single_projection(
    d_in: int, d_out: int, proj_type: str, use_layernorm: bool
) -> nn.Module:
    """
    Create a single projection layer.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        proj_type: "linear" or "mlp"
        use_layernorm: Apply LayerNorm before projection

    Returns:
        Projection module
    """
    layers = []

    # Optional LayerNorm
    if use_layernorm:
        layers.append(nn.LayerNorm(d_in))

    # Projection
    if proj_type == "linear":
        layers.append(nn.Linear(d_in, d_out, bias=False))

        # ##################### DEBUG ###################
        # # Use identity matrix, should produce aceptable outputs if comparing both final layers (11 in text is final).
        # with torch.no_grad():
        #     layers[-1].weight.copy_(torch.eye(d_in))
    elif proj_type == "mlp":
        layers.extend([nn.Linear(d_in, 1024), nn.GELU(), nn.Linear(1024, d_out)])
    else:
        raise ValueError(f"Unknown projection_type: {proj_type}")

    return nn.Sequential(*layers)


def _create_projection_model(ctx: TrainProjectionsContext) -> nn.Module:
    """
    Create projection model based on layer pair configuration.

    Only creates projections for intermediate layers (final layers are already
    in shared space). Returns a module that projects features and applies L2
    normalization.

    Returns:
        ProjectionModel that handles forward pass for both modalities
    """
    logger.info("Creating projection model...")

    # Determine which projections are needed
    text_is_final = ctx.text_layer == "final"
    image_is_final = ctx.image_layer == "final"

    # Both can't be final (nothing to train!)
    if text_is_final and image_is_final:
        raise ValueError("Both layers are 'final' - no projection needed!")

    # Determine input dimensions
    text_dim = 512  # Text features are always 512-dim (intermediate or final)
    image_dim = 768 if not image_is_final else 512  # Intermediate: 768, final: 512
    shared_dim = 512  # CLIP's shared embedding space

    # Create projection layers
    projections = {}

    if not text_is_final:
        logger.info(f"  Creating text projection: {text_dim} → {shared_dim}")
        projections["text"] = _create_single_projection(
            text_dim, shared_dim, ctx.projection_type, ctx.use_layernorm
        )
    else:
        logger.info("  Text layer is final (no projection needed)")
        projections["text"] = None

    if not image_is_final:
        logger.info(f"  Creating image projection: {image_dim} → {shared_dim}")
        projections["image"] = _create_single_projection(
            image_dim, shared_dim, ctx.projection_type, ctx.use_layernorm
        )
    else:
        logger.info("  Image layer is final (no projection needed)")
        projections["image"] = None

    # Wrap in a container module
    model = ProjectionWrapper(
        text_projection=projections["text"],
        image_projection=projections["image"],
        learnable_temperature=ctx.training_params["learnable_temperature"],
        initial_temperature=ctx.training_params["temperature"],
    )

    model = model.to(ctx.device)
    logger.info(f"✓ Projection model created on {ctx.device}")

    return model

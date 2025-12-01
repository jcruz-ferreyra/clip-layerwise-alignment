import logging

import torch
import torch.nn as nn

from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


def _create_optimizer(ctx: TrainProjectionsContext, model: nn.Module) -> torch.optim.Optimizer:
    """
    Create optimizer for projection training.

    Supports Adam and AdamW (CLIP uses AdamW with decoupled weight decay).

    Args:
        model: Model to optimize
        ctx: Training configuration

    Returns:
        Configured optimizer
    """
    optimizer_type = ctx.training_params.get("optimizer", "adamw").lower()
    lr = ctx.training_params["learning_rate"]
    wd = ctx.training_params["weight_decay"]

    logger.info(f"Creating optimizer: {optimizer_type.upper()}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Weight decay: {wd}")

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),  # CLIP defaults
            eps=1e-8,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}. Use 'adam' or 'adamw'.")

    logger.info("✓ Optimizer created")

    return optimizer

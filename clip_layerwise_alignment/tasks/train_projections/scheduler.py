import logging
from torch.optim import Optimizer
from torch.optim import lr_scheduler

from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


def _create_scheduler(ctx: TrainProjectionsContext, optimizer: Optimizer):
    """
    Create learning rate scheduler with warmup, plateau, and cosine decay.

    Schedule:
    1. Warmup: Linear ramp from start_factor * base_lr to base_lr
    2. Plateau: Constant at base_lr
    3. Cosine decay: Cosine annealing to min_lr

    Args:
        ctx: Training context
        optimizer: PyTorch optimizer

    Returns:
        lr_scheduler or None if scheduler disabled
    """
    # Check if scheduler is enabled
    use_scheduler = ctx.training_params.get("use_scheduler", True)
    if not use_scheduler:
        logger.info("Learning rate scheduler disabled")
        return None

    # Get params with defaults
    total_epochs = ctx.training_params["epochs"]  # Required
    warmup_epochs = ctx.training_params.get("warmup_epochs", 2)
    plateau_epochs = ctx.training_params.get("plateau_epochs", 3)
    warmup_start_factor = ctx.training_params.get("warmup_start_factor", 0.1)

    # Validate
    if warmup_epochs + plateau_epochs >= total_epochs:
        logger.warning(
            f"Warmup ({warmup_epochs}) + plateau ({plateau_epochs}) >= total epochs ({total_epochs})"
        )
        # Auto-adjust: warmup=25%, plateau=25%, cosine=50% of total epochs
        warmup_epochs = max(1, total_epochs // 4)
        plateau_epochs = max(1, total_epochs // 4)
        logger.warning(f"Auto-adjusted to: warmup={warmup_epochs}, plateau={plateau_epochs}")

    cosine_epochs = total_epochs - warmup_epochs - plateau_epochs

    logger.info("Learning rate scheduler: Warmup → Plateau → Cosine")
    logger.info(f"  Warmup: {warmup_epochs} epochs (start factor: {warmup_start_factor})")
    logger.info(f"  Plateau: {plateau_epochs} epochs")
    logger.info(f"  Cosine decay: {cosine_epochs} epochs")

    schedulers = []
    milestones = []

    # 1. Warmup phase
    if warmup_epochs > 0:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_epochs)

    # 2. Plateau phase (constant LR)
    if plateau_epochs > 0:
        plateau_scheduler = lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=plateau_epochs,
        )
        schedulers.append(plateau_scheduler)
        milestones.append(warmup_epochs + plateau_epochs)

    # 3. Cosine decay phase
    if cosine_epochs > 0:
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=0,  # Decay to 0 (can make configurable)
        )
        schedulers.append(cosine_scheduler)
        milestones.append(total_epochs)

    # Combine schedulers
    if len(schedulers) > 1:
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones[:-1],  # remove the last milestone
        )
    elif len(schedulers) == 1:
        scheduler = schedulers[0]
    else:
        logger.warning("No scheduler phases configured")
        return None

    return scheduler

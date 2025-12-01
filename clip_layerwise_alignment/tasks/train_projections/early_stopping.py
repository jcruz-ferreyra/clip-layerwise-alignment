import logging

from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """Check if should stop. Returns True to stop, False to continue."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping: no improvement for {self.patience} epochs")
                logger.info(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
                return True
            return False


def _create_early_stopping(ctx: TrainProjectionsContext):
    """
    Create early stopping callback for validation-based training.

    Stops training if validation loss doesn't improve for patience epochs.
    """
    patience = ctx.training_params.get("early_stopping_patience", 10)
    min_delta = ctx.training_params.get("early_stopping_min_delta", 1e-4)

    logger.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    return EarlyStopping(patience, min_delta)

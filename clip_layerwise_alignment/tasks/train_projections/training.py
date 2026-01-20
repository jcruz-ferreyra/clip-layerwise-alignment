import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .types import TrainProjectionsContext

logger = logging.getLogger(__name__)


def _run_training_loop(
    ctx: TrainProjectionsContext,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    """
    Run one training epoch.

    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        text_batch = batch["text"].to(ctx.device)                 # [N, 512]
        image_batch = batch["image"].to(ctx.device)               # [N, 768]

        # Forward
        text_embed, image_embed = model(text_batch, image_batch)  # [N, 512], [N, 512]

        # Contrastive loss
        logits = (image_embed @ text_embed.T) * model.temp        # [N, N] # Note that we used multiplying temperature parameter!!
        labels = torch.arange(len(text_batch), device=ctx.device) # [,N]

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def _run_val_loop(
    ctx: TrainProjectionsContext, model: nn.Module, val_loader: DataLoader, epoch: int
) -> float:
    """
    Run validation epoch (no gradient updates).

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch+1}"):
            text_batch = batch["text"].to(ctx.device)                 # [N, 512]
            image_batch = batch["image"].to(ctx.device)               # [N, 768]

            # Forward only
            text_embed, image_embed = model(text_batch, image_batch)  # [N, 512], [N, 512]

            # Loss
            logits = (image_embed @ text_embed.T) * model.temp        # [N, N]
            labels = torch.arange(len(text_batch), device=ctx.device) # [,N]

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

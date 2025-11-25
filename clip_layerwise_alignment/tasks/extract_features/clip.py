# tasks/extract_features/feature_extractors.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def _extract_vision_features(
    model: nn.Module, images: torch.Tensor, layer_indices: List[int]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract vision features at intermediate layers and final output.

    Vision CLS tokens are pre-extracted by forward_intermediates, making this
    straightforward. Intermediate features are 768-dim (ViT hidden size),
    final features are 512-dim (after projection to shared space).

    Args:
        model: CLIP model
        images: Image tensor [B, 3, 224, 224] (already on correct device)
        layer_indices: Layer indices to extract (0-indexed, e.g., [0, 2, 5, 8, 11])

    Returns:
        intermediate_features: Dict mapping layer names to features
            e.g., {'layer_1': [B, 768], 'layer_3': [B, 768], ...}
        final_features: [B, 512] - final projected embeddings (normalized)
    """
    with torch.no_grad():
        output = model.visual.forward_intermediates(
            images,
            indices=layer_indices,
            normalize_intermediates=False,  # Raw intermediate features
            output_extra_tokens=True,  # Get CLS tokens separately
            intermediates_only=False,  # Also return final output
            output_fmt="NLC",  # [B, N, C] format
        )

    # Extract CLS tokens from intermediates (already pooled for us!)
    cls_tokens = output["image_intermediates_prefix"]  # List of [B, 1, 768]

    intermediate_features = {
        f"layer_{layer_indices[i] + 1}": cls_tokens[i].squeeze(1)  # [B, 768]
        for i in range(len(layer_indices))
    }

    # Final features (already normalized and projected to 512-dim)
    final_features = output["image_features"]  # [B, 512]

    return intermediate_features, final_features


def _extract_text_features(
    model: nn.Module, text: torch.Tensor, layer_indices: List[int]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract text features at intermediate layers and final output.

    Unlike vision, text intermediates are full sequences [B, 77, 512] requiring
    manual [EOS] token extraction. [EOS] position varies per sample, so we use
    text.argmax() to locate it. All text features are 512-dim.

    Args:
        model: CLIP model (the full model, not just text encoder)
        text: Text token IDs [B, 77] (already on correct device)
        layer_indices: Layer indices to extract (0-indexed, e.g., [0, 2, 5, 8, 11])

    Returns:
        intermediate_features: Dict mapping layer names to features
            e.g., {'layer_1': [B, 512], 'layer_3': [B, 512], ...}
        final_features: [B, 512] - final projected embeddings (normalized)
    """
    B = text.shape[0]
    cast_dtype = model.transformer.get_cast_dtype()

    with torch.no_grad():
        # Embeddings
        x = model.token_embedding(text).to(cast_dtype)
        x = x + model.positional_embedding.to(cast_dtype)

        # Get attention mask (causal for standard CLIP)
        attn_mask = model.attn_mask if hasattr(model, "attn_mask") else None

        # Extract intermediates
        x_final, intermediates = model.transformer.forward_intermediates(
            x,
            attn_mask=attn_mask,
            indices=layer_indices,
            stop_early=False,
        )

    # Find [EOS] token positions (varies per sample)
    eos_positions = text.argmax(dim=-1)  # size: [B]
    batch_indices = torch.arange(B, device=text.device)  # tensor([0, 1, 2, ..., (B-1)]

    # Extract [EOS] tokens from each intermediate
    intermediate_features = {}
    for i, layer_idx in enumerate(layer_indices):
        eos_features = intermediates[i][batch_indices, eos_positions]  # [B, 512]
        intermediate_features[f"layer_{layer_idx + 1}"] = eos_features

    # Process final features: pool [EOS] → ln_final → projection
    x_final_pooled = x_final[batch_indices, eos_positions]  # [B, 512]
    x_final_pooled = model.ln_final(x_final_pooled)

    if model.text_projection is not None:
        if isinstance(model.text_projection, torch.nn.Linear):
            x_final_pooled = model.text_projection(x_final_pooled)
        else:
            x_final_pooled = x_final_pooled @ model.text_projection

    final_features = x_final_pooled  # [B, 512]

    return intermediate_features, final_features

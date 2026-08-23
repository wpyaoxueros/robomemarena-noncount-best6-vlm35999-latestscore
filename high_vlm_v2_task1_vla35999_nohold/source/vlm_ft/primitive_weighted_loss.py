"""Primitive-field weighted causal language-model loss.

The collator must provide ``primitive_token_mask`` aligned with ``labels``.
This intentionally weights only assistant tokens belonging to the JSON value of
``current_primitive``; prompt tokens and ignored labels remain unsupervised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def primitive_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    primitive_token_mask: torch.Tensor,
    primitive_weight: float = 1.0,
) -> torch.Tensor:
    if primitive_weight < 1.0:
        raise ValueError("primitive_weight must be >= 1")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = primitive_token_mask[..., 1:].to(dtype=torch.bool)
    per_token = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    weights = torch.ones_like(per_token)
    weights = torch.where(shift_mask & valid, weights * float(primitive_weight), weights)
    return (per_token * weights * valid).sum() / (weights * valid).sum().clamp_min(1.0)


class PrimitiveWeightedTrainerMixin:
    """Mixin for a Transformers Trainer whose collator emits the mask."""

    primitive_loss_weight: float = 1.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        inputs = dict(inputs)
        primitive_mask = inputs.pop("primitive_token_mask")
        outputs = model(**inputs, return_dict=True)
        loss = primitive_weighted_causal_lm_loss(
            outputs.logits, inputs["labels"], primitive_mask, self.primitive_loss_weight
        )
        return (loss, outputs) if return_outputs else loss

"""Special-token, progress-head, loss, and decoding contract for high_vlm_v2."""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

try:
    from .primitive_weighted_loss import primitive_weighted_causal_lm_loss
    from .training_components import QwenStrictTrajectoryCollator, decoded_substring_token_mask
except ImportError:
    from primitive_weighted_loss import primitive_weighted_causal_lm_loss
    from training_components import QwenStrictTrajectoryCollator, decoded_substring_token_mask


PROGRESS_TOKEN = "<|progress_query|>"
PROGRESS_HEAD_CONFIG_NAME = "high_vlm_v2_config.json"
DEFAULT_PROGRESS_LOSS_WEIGHT = 0.1
DEFAULT_SMOOTH_L1_BETA = 0.1


def register_progress_token(tokenizer) -> int:
    """Register one indivisible input-only query token and return its ID."""
    # Preserve Qwen's existing vision/control special-token registry. Replacing
    # it can silently alter skip_special_tokens decoding and multimodal parsing.
    special = {"additional_special_tokens": [PROGRESS_TOKEN]}
    try:
        # Transformers 5 names the option after ``extra_special_tokens``.
        tokenizer.add_special_tokens(special, replace_extra_special_tokens=False)
    except TypeError:
        # Transformers 4 compatibility.
        tokenizer.add_special_tokens(special, replace_additional_special_tokens=False)
    token_id = int(tokenizer.convert_tokens_to_ids(PROGRESS_TOKEN))
    encoded = tokenizer.encode(PROGRESS_TOKEN, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(f"{PROGRESS_TOKEN} is not represented by exactly one token: {encoded}")
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"{PROGRESS_TOKEN} unexpectedly maps to the unknown token")
    return token_id


def inject_progress_query(prompt_text: str, full_text: str | None = None) -> tuple[str, str | None]:
    """Put the query after the assistant header and before any answer token."""
    if PROGRESS_TOKEN in prompt_text:
        raise ValueError("generation prompt already contains the progress token")
    query_prompt = prompt_text + PROGRESS_TOKEN
    if full_text is None:
        return query_prompt, None
    if not full_text.startswith(prompt_text):
        raise ValueError("full chat serialization does not start with generation prompt")
    if PROGRESS_TOKEN in full_text:
        raise ValueError("full chat serialization already contains the progress token")
    query_full = query_prompt + full_text[len(prompt_text):]
    return query_prompt, query_full


def progress_token_position(input_ids: torch.Tensor, progress_token_id: int) -> int:
    positions = torch.nonzero(input_ids.eq(int(progress_token_id)), as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError(f"expected exactly one progress token, found {positions.numel()}")
    return int(positions.item())


def trim_generated_after_input(input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
    """Return generated continuation only; the input-only query is never decoded."""
    if input_ids.ndim != 1 or generated_ids.ndim != 1:
        raise ValueError("trim_generated_after_input expects unbatched token vectors")
    source_length = int(input_ids.numel())
    if generated_ids.numel() < source_length:
        raise ValueError("generated sequence is shorter than its input")
    if not torch.equal(generated_ids[:source_length].to(input_ids.device), input_ids):
        raise ValueError("generated sequence does not preserve the exact input prefix")
    return generated_ids[source_length:]


class ProgressHead(nn.Module):
    """Small scalar head over the final-norm hidden state at the query token."""

    def __init__(self, hidden_size: int, intermediate_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(hidden_size), int(intermediate_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(intermediate_size), 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_state).squeeze(-1)


def progress_prediction_and_loss(
    progress_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = DEFAULT_SMOOTH_L1_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    predictions = torch.sigmoid(progress_logits.float())
    targets = targets.to(device=predictions.device, dtype=torch.float32)
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("progress targets must lie in [0, 1]")
    loss = F.smooth_l1_loss(predictions, targets, beta=float(beta), reduction="mean")
    return predictions, loss


def _unwrap_causal_model(model: nn.Module) -> nn.Module:
    # Trainer passes DistributedDataParallel during multi-GPU training.
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def get_progress_head(model: nn.Module) -> nn.Module:
    base = _unwrap_causal_model(model)
    head = getattr(base, "progress_head", None)
    if head is None:
        raise AttributeError("model has no progress_head; attach it before PEFT loading")
    return head


def get_final_norm(model: nn.Module) -> nn.Module:
    base = _unwrap_causal_model(model)
    try:
        return base.model.language_model.norm
    except AttributeError as exc:
        raise AttributeError("cannot locate Qwen3-VL final language-model norm") from exc


def prepare_base_model_for_high_vlm_v2(
    model: nn.Module,
    tokenizer,
    config: dict[str, Any],
) -> int:
    """Resize a fresh base model and attach the head before PEFT restoration."""
    progress_id = register_progress_token(tokenizer)
    expected_id = config.get("query_token_id_at_training")
    if expected_id is not None and int(expected_id) != progress_id:
        raise ValueError(
            f"progress token ID mismatch: checkpoint={expected_id}, tokenizer={progress_id}"
        )
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    if embedding_rows != len(tokenizer):
        # Mean/covariance initialization performs a very large 4096x4096
        # covariance calculation independently on every rank.  A conventional
        # model initializer is sufficient because this one input row is trained.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    head_config = config["head"]
    hidden_size = int(head_config["input_size"])
    model_hidden_size = int(model.config.text_config.hidden_size)
    if hidden_size != model_hidden_size:
        raise ValueError(f"progress head/model hidden-size mismatch: {hidden_size} != {model_hidden_size}")
    model.progress_head = ProgressHead(
        hidden_size,
        intermediate_size=int(head_config["intermediate_size"]),
        dropout=float(head_config["dropout"]),
    ).to(device=model.device, dtype=model.dtype)
    return progress_id


def load_high_vlm_v2_adapter(
    base_model_path: str | Path,
    adapter_path: str | Path,
    *,
    model_load_kwargs: dict[str, Any] | None = None,
):
    """Load processor, resized base, progress head, and PEFT weights in the required order."""
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    adapter_path = Path(adapter_path)
    config_path = adapter_path / PROGRESS_HEAD_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True, local_files_only=True)
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        **(model_load_kwargs or {}),
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **kwargs)
    prepare_base_model_for_high_vlm_v2(model, processor.tokenizer, config)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, processor, config


class FinalNormQueryCapture(AbstractContextManager["FinalNormQueryCapture"]):
    """Capture only query-token states without retaining every transformer layer."""

    def __init__(self, model: nn.Module, positions: torch.Tensor) -> None:
        self.positions = positions.detach().to(dtype=torch.long)
        self.query_hidden: torch.Tensor | None = None
        self._handle = get_final_norm(model).register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        del module, args
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return
        if hidden.shape[0] != self.positions.numel():
            return
        positions = self.positions.to(hidden.device)
        if int(positions.max().item()) >= hidden.shape[1]:
            return
        # Generation calls the norm again for one-token decode steps.  Preserve
        # the first full-prefill capture, where the query positions exist.
        if self.query_hidden is None:
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            self.query_hidden = hidden[rows, positions]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._handle.remove()

    def require(self) -> torch.Tensor:
        if self.query_hidden is None:
            raise RuntimeError("final-norm hook did not observe every progress query position")
        return self.query_hidden


class HighVlmV2Collator(QwenStrictTrajectoryCollator):
    """Qwen collator with an input-only query token and scalar progress target."""

    def __init__(self, processor, max_length: int = 8192) -> None:
        self.progress_token_id = register_progress_token(processor.tokenizer)
        super().__init__(processor, max_length=max_length)

    def _encode(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        metadata = sample["metadata"]
        if "progress_target" not in metadata:
            raise ValueError(f"{sample['qid']}: missing progress_target")
        target = float(metadata["progress_target"])
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"{sample['qid']}: progress target is outside [0, 1]")

        messages = self._multimodal_messages(sample)
        prompt_messages = messages[:-1]
        plain_full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        plain_prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        query_prompt_text, query_full_text = inject_progress_query(plain_prompt_text, plain_full_text)
        assert query_full_text is not None
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs = dict(
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        full = self.processor(text=[query_full_text], **kwargs)
        prompt = self.processor(text=[query_prompt_text], **kwargs)
        input_ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"][0]
        prompt_length = int(prompt_ids.numel())
        if input_ids.numel() <= prompt_length:
            raise ValueError(f"{sample['qid']}: answer was truncated away")
        if not torch.equal(input_ids[:prompt_length], prompt_ids):
            raise ValueError(f"{sample['qid']}: processed prompt is not an exact prefix of full input")
        query_position = progress_token_position(input_ids, self.progress_token_id)
        if query_position != prompt_length - 1:
            raise ValueError(
                f"{sample['qid']}: query must be the final input token before the assistant answer; "
                f"position={query_position}, prompt_length={prompt_length}"
            )

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        if int(labels[query_position]) != -100:
            raise AssertionError(f"{sample['qid']}: input-only progress query is LM-supervised")
        supervised_positions = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
        if not supervised_positions.numel() or int(supervised_positions[0]) != query_position + 1:
            raise ValueError(f"{sample['qid']}: assistant answer does not begin immediately after query token")
        assistant_ids = input_ids[supervised_positions].tolist()
        assistant_text = self.tokenizer.decode(
            assistant_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if PROGRESS_TOKEN in assistant_text:
            raise ValueError(f"{sample['qid']}: progress token leaked into decoded assistant target")
        primitive = str(metadata["current_primitive"])
        local = decoded_substring_token_mask(self.tokenizer, assistant_ids, primitive)
        primitive_mask = torch.zeros_like(labels, dtype=torch.bool)
        primitive_mask[supervised_positions] = torch.tensor(local, dtype=torch.bool)
        if not primitive_mask.any():
            raise ValueError(f"{sample['qid']}: no primitive tokens selected")

        encoded = {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "primitive_token_mask": primitive_mask,
            "progress_position": torch.tensor(query_position, dtype=torch.long),
            "progress_target": torch.tensor(target, dtype=torch.float32),
        }
        for key in ("pixel_values", "image_grid_thw"):
            encoded[key] = full[key]
        return encoded

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [self._encode(sample) for sample in samples]
        return {
            "input_ids": pad_sequence(
                [x["input_ids"] for x in rows], batch_first=True, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": pad_sequence(
                [x["attention_mask"] for x in rows], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence([x["labels"] for x in rows], batch_first=True, padding_value=-100),
            "primitive_token_mask": pad_sequence(
                [x["primitive_token_mask"] for x in rows], batch_first=True, padding_value=False
            ),
            "progress_positions": torch.stack([x["progress_position"] for x in rows]),
            "progress_targets": torch.stack([x["progress_target"] for x in rows]),
            "pixel_values": torch.cat([x["pixel_values"] for x in rows], dim=0),
            "image_grid_thw": torch.cat([x["image_grid_thw"] for x in rows], dim=0),
        }


class HighVlmV2TrainerMixin:
    primitive_loss_weight: float = 1.0
    progress_loss_weight: float = DEFAULT_PROGRESS_LOSS_WEIGHT
    progress_smooth_l1_beta: float = DEFAULT_SMOOTH_L1_BETA
    _auxiliary_sums: dict[str, dict[str, float]] | None = None
    _auxiliary_counts: dict[str, int] | None = None

    def _accumulate_auxiliary(self, phase: str, values: dict[str, float]) -> None:
        if self._auxiliary_sums is None:
            self._auxiliary_sums = {"train": {}, "eval": {}}
            self._auxiliary_counts = {"train": 0, "eval": 0}
        sums = self._auxiliary_sums[phase]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        assert self._auxiliary_counts is not None
        self._auxiliary_counts[phase] += 1

    def _consume_auxiliary(self, phase: str, prefix: str = "") -> dict[str, float]:
        if self._auxiliary_sums is None or self._auxiliary_counts is None:
            return {}
        count = self._auxiliary_counts[phase]
        if count == 0:
            return {}
        result = {f"{prefix}{key}": value / count for key, value in self._auxiliary_sums[phase].items()}
        self._auxiliary_sums[phase] = {}
        self._auxiliary_counts[phase] = 0
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        inputs = dict(inputs)
        primitive_mask = inputs.pop("primitive_token_mask")
        progress_positions = inputs.pop("progress_positions")
        progress_targets = inputs.pop("progress_targets")
        labels = inputs.pop("labels")
        with FinalNormQueryCapture(model, progress_positions) as capture:
            outputs = model(**inputs, return_dict=True)
        query_hidden = capture.require()
        progress_logits = get_progress_head(model)(query_hidden)
        progress_predictions, progress_loss = progress_prediction_and_loss(
            progress_logits,
            progress_targets,
            beta=self.progress_smooth_l1_beta,
        )
        language_loss = primitive_weighted_causal_lm_loss(
            outputs.logits,
            labels,
            primitive_mask,
            self.primitive_loss_weight,
        )
        loss = language_loss + float(self.progress_loss_weight) * progress_loss
        self._accumulate_auxiliary("train" if model.training else "eval", {
            "lm_loss": float(language_loss.detach().float().item()),
            "progress_loss": float(progress_loss.detach().float().item()),
            "progress_mae": float(
                (progress_predictions - progress_targets.to(progress_predictions.device)).abs().mean().detach().item()
            ),
        })
        if return_outputs:
            # Keep Trainer prediction/eval compatibility without changing the
            # causal-LM output type expected by generation and checkpoints.
            return loss, outputs
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "eval_loss" in logs:
            logs = {**logs, **self._consume_auxiliary("eval", prefix="eval_")}
        elif "loss" in logs:
            logs = {**logs, **self._consume_auxiliary("train")}
        return super().log(logs, start_time)


def encode_generation_prompt(
    processor,
    messages: list[dict[str, Any]],
    *,
    images: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
    return_tensors: str = "pt",
) -> dict[str, torch.Tensor]:
    """Serialize inference input with the query as its final prompt token."""
    progress_id = register_progress_token(processor.tokenizer)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    query_prompt, _ = inject_progress_query(prompt_text)
    encoded = processor(
        text=[query_prompt],
        images=images,
        videos=videos,
        return_tensors=return_tensors,
        padding=True,
    )
    attention_mask = encoded.get("attention_mask")
    for row_index, row in enumerate(encoded["input_ids"]):
        position = progress_token_position(row, progress_id)
        if attention_mask is not None:
            non_padding = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
            if not non_padding.numel():
                raise ValueError("empty generation prompt")
            last_non_padding = int(non_padding[-1])
        else:
            last_non_padding = row.numel() - 1
        if position != last_non_padding:
            raise ValueError("progress query must be the final non-padding prompt token")
    return encoded


@torch.no_grad()
def generate_with_progress(
    model: nn.Module,
    processor,
    model_inputs: dict[str, torch.Tensor],
    **generation_kwargs: Any,
) -> tuple[list[str], torch.Tensor]:
    """Generate JSON and return progress while excluding the query from decode."""
    progress_id = int(processor.tokenizer.convert_tokens_to_ids(PROGRESS_TOKEN))
    input_ids = model_inputs["input_ids"]
    positions = torch.tensor(
        [progress_token_position(row, progress_id) for row in input_ids],
        device=input_ids.device,
        dtype=torch.long,
    )
    suppressed = list(generation_kwargs.pop("suppress_tokens", []) or [])
    if progress_id not in suppressed:
        suppressed.append(progress_id)
    generation_kwargs["suppress_tokens"] = suppressed
    with FinalNormQueryCapture(model, positions) as capture:
        generated = model.generate(**model_inputs, **generation_kwargs)
    predictions = torch.sigmoid(get_progress_head(model)(capture.require()).float())
    decoded: list[str] = []
    for source, complete in zip(input_ids, generated, strict=True):
        continuation = trim_generated_after_input(source, complete)
        text = processor.tokenizer.decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if PROGRESS_TOKEN in text:
            raise RuntimeError("input-only progress token leaked into decoded generation")
        decoded.append(text)
    return decoded, predictions

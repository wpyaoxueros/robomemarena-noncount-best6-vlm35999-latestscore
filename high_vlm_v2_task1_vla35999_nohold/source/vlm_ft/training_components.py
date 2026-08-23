"""Runnable HF components for the strict three-task Qwen3-VL dataset.

The collator marks the exact assistant tokens belonging to the value of
``current_primitive`` so experiments can optionally reweight that field.  The
default is ordinary 1:1 token loss for primitive and keyframe JSON tokens.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import Trainer

try:
    from .primitive_weighted_loss import PrimitiveWeightedTrainerMixin
except ImportError:  # Direct execution/tests from inside vlm_ft.
    from primitive_weighted_loss import PrimitiveWeightedTrainerMixin


def decoded_substring_token_mask(tokenizer, token_ids: Sequence[int], substring: str) -> list[bool]:
    ids = [int(x) for x in token_ids]
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    start = decoded.find(substring)
    if start < 0 or decoded.find(substring, start + 1) >= 0:
        raise ValueError(f"primitive substring is absent or ambiguous: {substring!r} in {decoded!r}")
    end = start + len(substring)
    offsets: list[tuple[int, int]] = []
    previous = ""
    for index in range(len(ids)):
        current = tokenizer.decode(ids[: index + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not current.startswith(previous):
            raise ValueError("non-monotonic tokenizer prefix decoding")
        offsets.append((len(previous), len(current)))
        previous = current
    return [right > start and left < end for left, right in offsets]


class JsonlOffsetDataset(Dataset):
    """Random-access JSONL without holding 32k multimodal records in RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        self.sampling_groups: list[tuple[int, str]] = []
        self.history_counts: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                self.offsets.append(offset)
                row = json.loads(line)
                metadata = row["metadata"]
                self.sampling_groups.append((int(metadata["task_id"]), str(metadata["sample_type"])))
                self.history_counts.append(int(metadata["history_count"]))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def hierarchical_balanced_sampler(self, *, seed: int = 0) -> WeightedRandomSampler:
        """Equalize the 3 tasks and 4 sample types without duplicating JSON/images."""
        counts = Counter(self.sampling_groups)
        weights = torch.tensor([1.0 / counts[group] for group in self.sampling_groups], dtype=torch.double)
        generator = torch.Generator().manual_seed(int(seed))
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


class QwenStrictTrajectoryCollator:
    def __init__(self, processor, max_length: int = 8192) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)

    def _multimodal_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        images = []
        for path in sample["images"]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        image_index = 0
        result: list[dict[str, Any]] = []
        for message in sample["messages"]:
            content: list[dict[str, Any]] = []
            parts = str(message["content"]).split("<image>")
            for part_index, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if part_index < len(parts) - 1:
                    content.append({"type": "image", "image": images[image_index]})
                    image_index += 1
            result.append({"role": message["role"], "content": content})
        if image_index != len(images):
            raise ValueError(f"{sample['qid']}: consumed {image_index}/{len(images)} images")
        return result

    def _encode(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        messages = self._multimodal_messages(sample)
        prompt_messages = messages[:-1]
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs = dict(
            images=image_inputs, videos=video_inputs, return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length,
        )
        full = self.processor(text=[full_text], **kwargs)
        # Image tokens precede the assistant answer, so this gives the exact
        # multimodal generation-prompt length using the identical processor.
        prompt = self.processor(text=[prompt_text], **kwargs)
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        labels[: int(prompt["input_ids"].shape[1])] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        supervised_positions = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
        assistant_ids = input_ids[supervised_positions].tolist()
        primitive = str(sample["metadata"]["current_primitive"])
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
        }
        for key in ("pixel_values", "image_grid_thw"):
            encoded[key] = full[key]
        return encoded

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [self._encode(sample) for sample in samples]
        result = {
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
            "pixel_values": torch.cat([x["pixel_values"] for x in rows], dim=0),
            "image_grid_thw": torch.cat([x["image_grid_thw"] for x in rows], dim=0),
        }
        return result


class PrimitiveWeightedTrainer(PrimitiveWeightedTrainerMixin, Trainer):
    primitive_loss_weight = 1.0

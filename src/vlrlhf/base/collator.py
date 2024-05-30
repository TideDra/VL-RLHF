from typing import Any, List, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
import torch
from dataclasses import dataclass
from abc import ABC


@dataclass
class VLDPODataCollatorWithPadding(ABC):
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False
    processor: Optional[Any] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


@dataclass
class VLSFTDataCollatorWithPadding(ABC):
    pad_token_id: int
    label_pad_token_id: int
    processor: Optional[Any] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            if k == "img_path":
                padded_batch[k] = [ex[k] for ex in features]
                continue
            to_pad = [torch.LongTensor(ex[k]) for ex in features]
            if k == "input_ids":
                padding_value = self.pad_token_id
            elif k == "labels":
                padding_value = self.label_pad_token_id
            elif k == "attention_mask":
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
        return padded_batch


@dataclass
class VLRMDataCollatorWithPadding(ABC):
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            to_pad = [torch.LongTensor(ex[k]) for ex in features]
            if k.startswith("input_ids"):
                padding_value = self.pad_token_id
            elif k.startswith("attention_mask"):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
        return padded_batch


@dataclass
class VLPPODataCollator(ABC):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            for feature in features:
                data = feature[key]
                if key in ["input_ids", "attention_mask"]:
                    data = torch.LongTensor(data)
                if key not in batch:
                    batch[key] = []
                batch[key].append(data)
        return batch

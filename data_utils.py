from datasets import load_dataset
import torch
from collections import defaultdict
from itertools import combinations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
def make_vlfeedback_paired_dataset(local_rank,cache_dir):
    ds = load_dataset("MMInstruction/VLFeedback", split="train",cache_dir=cache_dir,trust_remote_code=True)

    # make comparison pairs from completion list
    if local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, comps in enumerate(sample["completions"]):
            prompt = sample["prompt"][sample_idx]
            prompt = "USER: <image>\n"+prompt+" ASSISTANT:"
            img_path = sample['img_path'][sample_idx]
            for comp_idx1, comp_idx2 in combinations(range(len(comps["annotations"])), 2):
                anno1, anno2 = comps["annotations"][comp_idx1], comps["annotations"][comp_idx2]

                # get average scores
                try:
                    avg_score1 = np.mean(
                        [
                            float(anno1[aspect]["Rating"])
                            for aspect in anno1
                        ]
                    )
                    avg_score2 = np.mean(
                        [
                            float(anno2[aspect]["Rating"])
                            for aspect in anno2
                        ]
                    )
                except ValueError:
                    continue

                # get chosen and rejected responses
                if avg_score1 > avg_score2:
                    chosen = comps["response"][comp_idx1]
                    rejected = comps["response"][comp_idx2]
                elif avg_score2 > avg_score1:
                    chosen = comps["response"][comp_idx2]
                    rejected = comps["response"][comp_idx1]
                else:
                    continue
                converted_sample["prompt"].append(prompt)
                converted_sample["chosen"].append(chosen)
                converted_sample["rejected"].append(rejected)
                converted_sample["img_path"].append(img_path)

        return converted_sample

    ds = ds.map(
        make_batch_pairs,
        batched=True,
        remove_columns=set(ds.column_names) - set(["prompt", "chosen", "rejected","img_path"]),
        keep_in_memory=True
    )

    if local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    return ds

@dataclass
class LLaVADPODataCollatorWithPadding:
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
    image_processor: Any = None
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
            elif k == 'img_path':
                imgs = [Image.open(ex[k]) for ex in features]
                pixel_values = self.image_processor(imgs, return_tensors='pt')["pixel_values"].half()
                padded_batch['pixel_values'] = pixel_values
            else: 
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch
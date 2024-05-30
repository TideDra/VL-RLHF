from datasets import load_dataset
from collections import defaultdict
from itertools import combinations
import numpy as np
from datasets import Dataset
import json
import os
from accelerate import PartialState


def make_vlfeedback_paired_dataset(script_args):
    score_margin = script_args.score_margin
    ds = load_dataset("MMInstruction/VLFeedback", split="train", trust_remote_code=True)

    # make comparison pairs from completion list
    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, comps in enumerate(sample["completions"]):
            prompt = sample["prompt"][sample_idx]
            img_path = sample["img_path"][sample_idx]

            temp = defaultdict(lambda: defaultdict(list))
            for comp_idx1, comp_idx2 in combinations(range(len(comps["annotations"])), 2):
                anno1, anno2 = (
                    comps["annotations"][comp_idx1],
                    comps["annotations"][comp_idx2],
                )

                # get average scores
                try:
                    avg_score1 = np.mean([float(anno1[aspect]["Rating"]) for aspect in anno1])
                    avg_score2 = np.mean([float(anno2[aspect]["Rating"]) for aspect in anno2])
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
                score_gap = abs(avg_score1 - avg_score2)

                temp[score_gap]["prompt"].append(prompt)
                temp[score_gap]["chosen"].append(chosen)
                temp[score_gap]["rejected"].append(rejected)
                temp[score_gap]["img_path"].append(img_path)

            if len(temp) == 0:
                continue

            if score_margin == -1:
                # select pairs with the largest score gap
                max_score_gap = max(temp.keys())
                converted_sample["prompt"].extend(temp[max_score_gap]["prompt"])
                converted_sample["chosen"].extend(temp[max_score_gap]["chosen"])
                converted_sample["rejected"].extend(temp[max_score_gap]["rejected"])
                converted_sample["img_path"].extend(temp[max_score_gap]["img_path"])
            else:
                # select pairs with the score gap larger than the score margin
                for score_gap in temp.keys():
                    if score_gap >= score_margin:
                        converted_sample["prompt"].extend(temp[score_gap]["prompt"])
                        converted_sample["chosen"].extend(temp[score_gap]["chosen"])
                        converted_sample["rejected"].extend(temp[score_gap]["rejected"])
                        converted_sample["img_path"].extend(temp[score_gap]["img_path"])

        return converted_sample

    with PartialState().local_main_process_first():
        ds = ds.map(
            make_batch_pairs,
            batched=True,
            remove_columns=set(ds.column_names) - set(["prompt", "chosen", "rejected", "img_path"]),
            keep_in_memory=True,
        )

    return ds


def build_dataset_from_vlquery_json(script_args):
    json_path = script_args.data_path
    image_root = script_args.image_root
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    def gen():
        for data in raw_data:
            data["img_path"] = os.path.join(image_root, data["image"])
            yield data

    with PartialState().local_main_process_first():
        dataset = Dataset.from_generator(gen)
    return dataset


def make_rlhfv_paired_dataset(script_args):
    ds = load_dataset("HaoyeZhang/RLHF-V-Dataset", split="train", trust_remote_code=True)

    def preprocess(sample):
        sample["img_path"] = os.path.join(script_args.image_root, sample["image_path"])
        text = json.loads(sample["text"])
        sample["prompt"] = text["question"]
        sample["chosen"] = text["chosen"]
        sample["rejected"] = text["rejected"]
        return sample

    with PartialState().local_main_process_first():
        ds = ds.map(
            preprocess,
            remove_columns=set(ds.column_names) - set(["prompt", "chosen", "rejected", "img_path"]),
        )
    return ds


def build_plain_dpo_dataset(script_args):
    image_root = script_args.image_root
    json_path = script_args.data_path
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    def gen():
        for d in raw_data:
            prompt = d["prompt"]
            chosen = d["chosen"]
            rejected = d["rejected"]
            if "image" in d:
                img_path = os.path.join(image_root, d["image"])
            else:
                img_path = None
            yield dict(img_path=img_path, prompt=prompt, chosen=chosen, rejected=rejected)

    with PartialState().local_main_process_first():
        ds = Dataset.from_generator(gen)
    return ds


DATASET_MAP = {
    "vlfeedback_paired": make_vlfeedback_paired_dataset,
    "vlquery_json": build_dataset_from_vlquery_json,
    "rlhfv": make_rlhfv_paired_dataset,
    "plain_dpo": build_plain_dpo_dataset,
}

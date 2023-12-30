from datasets import load_dataset
import torch
from collections import defaultdict
from itertools import combinations
import numpy as np

def make_vlfeedback_paired_dataset(local_rank:int,cache_dir:str,score_margin:float = -1):
    ds = load_dataset("MMInstruction/VLFeedback", split="train",cache_dir=cache_dir,trust_remote_code=True)

    # make comparison pairs from completion list
    if local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, comps in enumerate(sample["completions"]):
            prompt = sample["prompt"][sample_idx]
            img_path = sample['img_path'][sample_idx]
            
            temp = defaultdict(lambda:defaultdict(list))
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
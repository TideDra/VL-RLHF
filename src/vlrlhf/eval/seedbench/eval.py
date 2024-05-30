from accelerate import Accelerator
from torch.utils.data import Dataset
import os
import argparse
import json
from ..utils import run_vqa_ppl, VLCollator
from copy import deepcopy

accelerator = Accelerator(mixed_precision="bf16")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default="seedbench_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--processor_path", type=str, default=None)

    return parser.parse_args()


class SeedbenchDataset(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, "SEED-Bench.json"), "r") as f:
            self.raw_data = json.load(f)["questions"]
        self.data = []
        for q in self.raw_data:
            if q["question_type_id"] > 9:
                # filter out video
                continue
            sample = {
                "question_id": q["question_id"],
                "image": os.path.join(data_root, "SEED-Bench-image", q["data_id"]),
                "prompt": q["question"],
            }
            sample["response"] = "The answer is: " + q["choice_a"]
            sample["choice"] = "A"
            self.data.append(deepcopy(sample))
            sample["response"] = "The answer is: " + q["choice_b"]
            sample["choice"] = "B"
            self.data.append(deepcopy(sample))
            sample["response"] = "The answer is: " + q["choice_c"]
            sample["choice"] = "C"
            self.data.append(deepcopy(sample))
            sample["response"] = "The answer is: " + q["choice_d"]
            sample["choice"] = "D"
            self.data.append(deepcopy(sample))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator(VLCollator):
    def __call__(self, batch):
        question_ids = [b["question_id"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        images = [b["image"] for b in batch]
        responses = [b["response"] for b in batch]
        choices = [b["choice"] for b in batch]
        prompts = [self.processor.format_multimodal_prompt(q, img) for q, img in zip(prompts, images)]
        convs = [self.processor.make_single_turn_conv(prompt, res) for prompt, res in zip(prompts, responses)]
        inputs = self.processor(convs=convs, images_path=images, padding_side="right")
        others = [dict(question_id=id, choice=choice) for id, choice in zip(question_ids, choices)]
        return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    dataset = SeedbenchDataset(data_root)
    results = run_vqa_ppl(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)
    final_answers = {}
    for r in results:
        question_id = r["question_id"]
        choice = r["choice"]
        ppl = r["ppl"]
        if question_id not in final_answers:
            final_answers[question_id] = {"choice": choice, "ppl": ppl}
        else:
            if ppl < final_answers[question_id]["ppl"]:
                final_answers[question_id] = {"choice": choice, "ppl": ppl}
    with open(args.output_path, "w") as f:
        json.dump(final_answers, f)

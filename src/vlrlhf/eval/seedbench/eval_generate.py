from accelerate import Accelerator
from torch.utils.data import Dataset
import os
import argparse
import json
from ..utils import run_vqa, VLCollator

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
                "question": q["question"],
                "prompt": f"{q['question']}\n Here are four options for you to choose from:\nA. {q['choice_a']}\nB. {q['choice_b']}\nC. {q['choice_c']}\nD. {q['choice_d']}\nPlease only output the option letter.",  # noqa:E501
                "choice_a": q["choice_a"],
                "choice_b": q["choice_b"],
                "choice_c": q["choice_c"],
                "choice_d": q["choice_d"],
            }
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator(VLCollator):
    def __call__(self, batch):
        prompts = [b["prompt"] for b in batch]
        images = [b["image"] for b in batch]
        prompts = [self.processor.format_multimodal_prompt(q, img) for q, img in zip(prompts, images)]
        inputs = self.processor(texts=prompts, images_path=images, padding_side="left")
        others = batch
        return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    dataset = SeedbenchDataset(data_root)
    results = run_vqa(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)
    with open(args.output_path, "w") as f:
        json.dump(results, f)

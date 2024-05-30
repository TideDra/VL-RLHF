from torch.utils.data import Dataset
import os
import argparse
import json
from ..utils import run_vqa_sgl, get_model_cache
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_dir/mm-vet")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=64)
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
                "prompt": f"""{q['question']}\n Here are four options for you to choose from:\nA. {q['choice_a']}\nB. {q['choice_b']}\nC. {q['choice_c']}\nD. {q['choice_d']}\nPlease only output the option letter.""",  # noqa:E501
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


def collator(batch):
    questions = [b["prompt"] for b in batch]
    images = [b["image"] for b in batch]
    inputs = [{"image_path": i, "question": q} for i, q in zip(images, questions)]
    others = batch
    return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    dataset = SeedbenchDataset(data_root)
    results = run_vqa_sgl(
        model_path=model_path,
        dataset=dataset,
        collator=collator,
        processor_path=args.processor_path,
        batch_size=args.batch_size,
        endpoint=args.endpoint,
    )
    final_answers = {}
    for r in results:
        question_id = r["question_id"]
        response = r["response"]
        if len(response) == 0:
            choice = random.choice(["A", "B", "C", "D"])
        elif response[0] in ["A", "B", "C", "D"]:
            choice = response[0]
        elif response == r["choice_a"]:
            choice = "A"
        elif response == r["choice_b"]:
            choice = "B"
        elif response == r["choice_c"]:
            choice = "C"
        elif response == r["choice_d"]:
            choice = "D"
        else:
            choice = random.choice(["A", "B", "C", "D"])
        final_answers[question_id] = {"choice": choice, "response": response}

    with open(args.output_path, "w") as f:
        json.dump(final_answers, f)
    model_cache = get_model_cache()
    for v in model_cache.values():
        v.shutdown()

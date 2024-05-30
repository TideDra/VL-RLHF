import json
from accelerate import Accelerator
import argparse
from torch.utils.data import Dataset
from ..utils import run_vqa, VLCollator
import os

accelerator = Accelerator(mixed_precision="bf16")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/gozhang/data_dir/mm-vet")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="mmvet_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


class MMVetDataset(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, "mm-vet.json"), "r") as f:
            data = json.load(f)
        self.data = [(k, v) for k, v in data.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "id": self.data[index][0],
            "image": os.path.join(self.data_root, "images", self.data[index][1]["imagename"]),
            "question": self.data[index][1]["question"],
            "answer": self.data[index][1]["answer"],
            "category": ",".join(self.data[index][1]["capability"]),
        }


class Collator(VLCollator):
    def __call__(self, batch):
        ids = [b["id"] for b in batch]
        categories = [b["category"] for b in batch]
        answers = [b["answer"] for b in batch]
        questions = [b["question"] for b in batch]
        images = [b["image"] for b in batch]
        prompt = [
            self.processor.format_multimodal_prompt(q, img).replace("Picture 1: ", "")
            for q, img in zip(questions, images)
        ]
        inputs = self.processor(texts=prompt, images_path=images, padding_side="left", check_format=False)
        others = [
            dict(index=index, answer=answer, question=question, category=category)
            for index, answer, question, category in zip(ids, answers, questions, categories)
        ]
        return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    dataset = MMVetDataset(data_root)
    results = run_vqa(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(dict((r["index"], r["prediction"]) for r in results), f, indent=4)

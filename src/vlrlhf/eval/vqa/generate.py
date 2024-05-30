from accelerate import Accelerator
import json
from torch.utils.data import Dataset
import os
import argparse
from ..utils import run_vqa, VLCollator

accelerator = Accelerator(mixed_precision="bf16")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="vqa_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


class VQADataset(Dataset):
    def __init__(self, data_root, file_path) -> None:
        super().__init__()
        self.data_root = data_root
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"image": os.path.join(self.data_root, self.data[index]["image"]), "prompt": self.data[index]["prompt"]}


class Collator(VLCollator):
    def __call__(self, batch):
        questions = [b["prompt"] for b in batch]
        images = [b["image"] for b in batch]
        prompt = [self.processor.format_multimodal_prompt(q, img) for q, img in zip(questions, images)]
        inputs = self.processor(texts=prompt, images_path=images, padding_side="left", check_format=False)
        others = [{"image": img, "prompt": p} for img, p in zip(images, questions)]
        return inputs, others


if __name__ == "__main__":

    args = parse_args()
    data_root = args.data_root
    file_path = args.file_path

    dataset = VQADataset(data_root, file_path)
    results = run_vqa(args.model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)

    if accelerator.is_local_main_process:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)

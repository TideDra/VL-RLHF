import json
from torch.utils.data import Dataset
import os
import argparse
from ..utils import run_vqa_sgl, get_model_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_dir/mm-vet")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=64)
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


def collator(batch):
    ids = [b["id"] for b in batch]
    questions = [b["question"] for b in batch]
    images = [b["image"] for b in batch]
    inputs = [{"image_path": i, "question": q} for i, q in zip(images, questions)]
    others = [dict(index=id) for id in ids]
    return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    processor_path = args.processor_path
    batch_size = args.batch_size
    endpoint = args.endpoint
    dataset = MMVetDataset(data_root)
    results = run_vqa_sgl(
        model_path=model_path,
        dataset=dataset,
        collator=collator,
        processor_path=processor_path,
        batch_size=batch_size,
        endpoint=endpoint,
    )
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(dict((r["index"], r["prediction"]) for r in results), f, indent=4)
    model_cache = get_model_cache()
    for v in model_cache.values():
        v.shutdown()

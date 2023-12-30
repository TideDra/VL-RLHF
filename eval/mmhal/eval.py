import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/gozhang/code/LLaVA/playground/data/eval/mmhal",
    )
    parser.add_argument(
        "--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf"
    )
    parser.add_argument("--output_path", type=str, default="mmvet_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


class MMHalDataset(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, "response_template.json"), "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        imagename = data["image_src"].split("/")[-1]
        imagename = os.path.join(self.data_root, "images", imagename)
        image = Image.open(imagename).convert("RGB")
        data["image"] = image
        data["prompt"] = "USER: <image>\n" + data["question"] + " ASSISTANT:"
        return self.data[index]


def collator(batch):
    concat_batch = defaultdict(list)
    for data in batch:
        for key, item in data.items():
            concat_batch[key].append(item)
    concat_batch["input"] = processor(
        text=concat_batch["prompt"],
        images=concat_batch["image"],
        return_tensors="pt",
        padding=True,
    )
    return concat_batch


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor.tokenizer.pad_token = processor.tokenizer.bos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    )
    model.to("cuda")
    dataset = MMHalDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch["input"]
            inputs.to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].half()
            outputs = model.generate(
                **inputs,
                temperature=args.temperature,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
            input_token_len = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, input_token_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for (
                question_type,
                question_topic,
                image_id,
                image_src,
                image_content,
                question,
                gt_answer,
                model_answer,
            ) in zip(
                batch["question_type"],
                batch["question_topic"],
                batch["image_id"],
                batch["image_src"],
                batch["image_content"],
                batch["question"],
                batch["gt_answer"],
                responses,
            ):
                results.append(
                    {
                        "question_type": question_type,
                        "question_topic": question_topic,
                        "image_id": image_id,
                        "image_src": image_src,
                        "image_content": image_content,
                        "question": question,
                        "gt_answer": gt_answer,
                        "model_answer": model_answer,
                    }
                )
            bar.update(len(responses))
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

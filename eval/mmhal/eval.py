import json
from PIL import Image
from utils.auto_load import MyAutoModel, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import copy
from peft import PeftModel
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
        data["image"] = imagename
        data["prompt"] = copy.deepcopy(data["question"])
        return data


def collator(batch):
    concat_batch = defaultdict(list)
    for data in batch:
        for key, item in data.items():
            concat_batch[key].append(item)
    imgs_path = concat_batch["image"]
    concat_batch['prompt'] = [processor.format_multimodal_prompt(prompt,img) for prompt,img in zip(concat_batch['prompt'],imgs_path)]
    inputs = processor(texts=concat_batch['prompt'],images_path=imgs_path,padding_side='left')
    concat_batch['input'] = inputs
    return concat_batch


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    model = MyAutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16)
    if isinstance(model,PeftModel):
        model_path = model.peft_config['default'].base_model_name_or_path
    processor = MyAutoProcessor.from_pretrained(model_path)
    processor.infer()
    tokenizer = processor.tokenizer

    model.to('cuda')
    dataset = MMHalDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(model_path)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch["input"]
            inputs.to("cuda")
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
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

import json
from PIL import Image
from utils.auto_load import MyAutoModelWithValueHead, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import copy
from peft import LoraConfig
import wandb
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
    parser.add_argument("--reward_adapter", type=str, default=None)
    parser.add_argument("--best_of", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--project_name", type=str, default="VL-RLHF")
    parser.add_argument("--group_name", type=str, default="Qwen-VL-rm-eval")
    parser.add_argument("--run_name", type=str, default=None)
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
    run = wandb.init(project=args.project_name,group=args.group_name,name=args.run_name)
    data_root = args.data_root
    processor = MyAutoProcessor.from_pretrained(args.model_path)
    processor.infer()
    tokenizer = processor.tokenizer
    model = MyAutoModelWithValueHead.from_pretrained(
        args.model_path, torch_dtype=torch.float16,peft_config=LoraConfig.from_pretrained(args.reward_adapter),reward_adapter=args.reward_adapter
    )
    model.to("cuda")
    dataset = MMHalDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(args.model_path)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    rows = []
    with torch.inference_mode():
        for idx,batch in enumerate(dataloader):
            inputs = batch["input"]
            inputs.to("cuda")
            input_ids = inputs.input_ids.repeat(args.best_of,1)
            attention_mask = inputs.attention_mask.repeat(args.best_of,1)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                temperature=args.temperature,
                use_cache=True
            )

            input_token_len = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, input_token_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            #### Compute reward score
            convs = [processor.make_single_turn_conv(q,r) for q, r in zip(batch["prompt"]*args.best_of, responses)]

            inputs = processor.process_batch_conv(convs)['full']
            inputs.pop("labels")
            padding_side_default = tokenizer.padding_side
            tokenizer.padding_side = "left"
            padded_inputs = tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                return_tensors="pt",
            ).to('cuda')
            tokenizer.padding_side = padding_side_default
            rewards = model.compute_reward_score(**padded_inputs).squeeze(-1)
            best_idx = rewards.argmax()
            best_reward = rewards[best_idx].item()
            best_response = responses[best_idx]
            worst_idx = rewards.argmin()
            worst_response = responses[worst_idx]
            worst_reward = rewards[worst_idx].item()
            rewards = rewards.to(torch.float).cpu().numpy()
            row = [wandb.Image(batch['image'][0]),batch['question'][0],best_response,best_reward,worst_response,worst_reward]
            rows.append(row)
            wandb.log({"reward":rewards})

            bar.update(1)
    table = wandb.Table(columns=["image","prompt","best_response","best_reward","worst_response","worst_reward"],rows=rows)
    wandb.log({"game_log":table})
    run.finish()

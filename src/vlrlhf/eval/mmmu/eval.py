from PIL import Image
from accelerate import Accelerator
from ..utils import run_vqa, VLCollator
from torch.utils.data import Dataset
import argparse
import base64
import pandas as pd
import io
from collections import defaultdict
import tempfile
import numpy as np
import os
import string
import re

accelerator = Accelerator(mixed_precision="bf16")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default="mmmu_result.xlsx")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--processor_path", type=str, default=None)
    return parser.parse_args()


class MMMUDataset(Dataset):

    def __init__(self, data_file, sys_prompt="There are several options:"):

        def isliststr(s):
            return (s[0] == "[") and (s[-1] == "]")

        def istype(s, type):
            if isinstance(s, type):
                return True
            try:
                return isinstance(eval(s), type)
            except Exception:
                return False

        data = pd.read_csv(data_file, sep="\t")
        data = data[~pd.isna(data["image"])]
        data["index"] = [str(x) for x in data["index"]]
        data["image"] = [str(x) for x in data["image"]]

        image_map = {x: y for x, y in zip(data["index"], data["image"])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        data["image"] = [eval(image_map[k]) if isliststr(image_map[k]) else image_map[k] for k in data["index"]]
        if "image_path" in data:
            data["image_path"] = [eval(pths) if isliststr(pths) else pths for pths in data["image_path"]]
        if np.all([istype(x, int) for x in data["index"]]):
            data["index"] = [int(x) for x in data["index"]]

        self.df = data
        self.sys_prompt = sys_prompt
        self.temp_img_dir = tempfile.TemporaryDirectory()
        self.__img_id = 0

    def dump_image_to_tempfile(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        name = os.path.join(self.temp_img_dir.name, f"{self.__img_id}.jpg")
        self.__img_id += 1
        image.save(name)
        return name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        image = line["image"]
        question = line["question"]
        if isinstance(image, str):
            image = [self.dump_image_to_tempfile(image)]
        elif isinstance(image, list):
            image = [self.dump_image_to_tempfile(img) for img in image]

        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += "Please select the correct answer from the options above. \n"
        used_image_idx = re.findall(r"<image (\d)>", prompt, re.S)
        used_image_idx = [int(i) - 1 for i in used_image_idx]
        data = {"img": [image[i] for i in used_image_idx], "prompt": re.sub(r"<image \d>", "<image>", prompt, re.S)}
        data.update(line.to_dict())
        data.pop("image")
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class Collator(VLCollator):
    def __call__(self, batch):
        images = [b["img"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        prompts = [self.processor.format_multimodal_prompt(prompt, img) for prompt, img in zip(prompts, images)]
        inputs = self.processor(texts=prompts, images_path=images, padding_side="left")
        others = batch
        return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    dataset = MMMUDataset(data_root)
    results = run_vqa(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)
    results = [r for r in results if r.update(response=r.pop("response")) is None]
    dataset.temp_img_dir.cleanup()
    answer_upload = defaultdict(list)

    for r in results:
        response = r.pop("response")
        answer_upload["prediction"].append(response)
        for k, v in r.items():
            answer_upload[k].append(v)
    answer_upload = pd.DataFrame(answer_upload)
    answer_upload.to_excel(args.output_path, index=False)

from torch.utils.data import Dataset
import os
import argparse
from ..utils import run_vqa_sgl, get_model_cache
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
import base64
import io
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_dir/MME")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


class MMEDataset(Dataset):

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

        if isinstance(image, str):
            image = self.dump_image_to_tempfile(image)
        elif isinstance(image, list):
            image = [self.dump_image_to_tempfile(img) for img in image]

        data = line.to_dict()
        data["img"] = image
        data.pop("image")
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


def collator(batch):
    inputs = [dict(image_path=b["img"], question=b["question"]) for b in batch]
    others = batch
    return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    processor_path = args.processor_path
    batch_size = args.batch_size
    endpoint = args.endpoint
    dataset = MMEDataset(data_root)
    results = run_vqa_sgl(
        model_path=model_path,
        dataset=dataset,
        collator=collator,
        processor_path=processor_path,
        batch_size=batch_size,
        endpoint=endpoint,
    )
    dataset.temp_img_dir.cleanup()
    answer_upload = defaultdict(list)

    for r in results:
        response = r.pop("response")
        answer_upload["prediction"].append(response)
        for k, v in r.items():
            answer_upload[k].append(v)
    answer_upload = pd.DataFrame(answer_upload)
    answer_upload.to_excel(args.output_path, index=False)
    model_cache = get_model_cache()
    for v in model_cache.values():
        v.shutdown()

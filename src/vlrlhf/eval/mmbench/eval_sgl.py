from PIL import Image
from ..utils import run_vqa_sgl, get_model_cache
from torch.utils.data import Dataset
import argparse
import base64
import pandas as pd
import io
from collections import defaultdict
import tempfile
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_dir/MME")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


class MMBenchDataset(Dataset):
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
        index = self.df.iloc[idx]["index"]
        image = self.df.iloc[idx]["image"]
        image = self.dump_image_to_tempfile(image)
        question = self.df.iloc[idx]["question"]
        answer = self.df.iloc[idx]["answer"] if "answer" in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]["category"]
        l2_catetory = self.df.iloc[idx]["l2-category"]
        option_candidate = ["A", "B", "C", "D", "E"]
        options = {
            cand: self.load_from_df(idx, cand) for cand in option_candidate if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = self.load_from_df(idx, "hint")
        data = {
            "img": image,
            "question": question,
            "answer": answer,
            "options": options_prompt,
            "category": catetory,
            "l2-category": l2_catetory,
            "options_dict": options,
            "index": index,
            "context": hint,
        }
        if data["context"] is not None:
            prompt = (
                data["context"]
                + " "
                + data["question"]
                + " "
                + data["options"]
                + "\n"
                + "please only output the option letter."
            )
        else:
            prompt = data["question"] + " " + data["options"] + "\n" + "please only output the option letter."

        data["prompt"] = prompt
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


def collator(batch):
    inputs = [dict(image_path=b["img"], question=b["prompt"]) for b in batch]
    others = batch
    return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    processor_path = args.processor_path
    batch_size = args.batch_size
    endpoint = args.endpoint
    dataset = MMBenchDataset(data_root)
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
        question = r["question"]
        options_dict = r["options_dict"]
        category = r["category"]
        l2_category = r["l2-category"]
        index = r["index"]
        response = r["response"]
        answer = r["answer"]
        choice_A = options_dict.get("A", "")
        choice_B = options_dict.get("B", "")
        choice_C = options_dict.get("C", "")
        choice_D = options_dict.get("D", "")
        split = "dev" if "dev" in data_root.lower() else "test"
        answer_upload["question"].append(question)
        answer_upload["A"].append(choice_A)
        answer_upload["B"].append(choice_B)
        answer_upload["C"].append(choice_C)
        answer_upload["D"].append(choice_D)
        answer_upload["prediction"].append(response)
        answer_upload["category"].append(category)
        answer_upload["l2_category"].append(l2_category)
        answer_upload["index"].append(index)
        answer_upload["split"].append(split)
        answer_upload["answer"].append(answer)
    answer_upload = pd.DataFrame(answer_upload)
    answer_upload.to_excel(args.output_path, index=False)
    model_cache = get_model_cache()
    for v in model_cache.values():
        v.shutdown()

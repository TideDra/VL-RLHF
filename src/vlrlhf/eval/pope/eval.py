from accelerate import Accelerator
import json
from torch.utils.data import Dataset
import os
import argparse
from ..utils import run_vqa, VLCollator, log_data_to_mysql
from loguru import logger

accelerator = Accelerator(mixed_precision="bf16")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--file_root", type=str)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="pope_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--report_to_mysql", type=bool, default=False)
    parser.add_argument("--sql_host", type=str, default=None)
    parser.add_argument("--sql_port", type=int, default=3306)
    parser.add_argument("--sql_user", type=str, default=None)
    parser.add_argument("--sql_password", type=str, default=None)
    parser.add_argument("--sql_db", type=str, default=None)
    parser.add_argument("--sql_table", type=str, default="POPE")
    parser.add_argument("--sql_tag", type=str, default=None)
    return parser.parse_args()


def evaluate(outputs):
    answers = [dict(answer=o["response"]) for o in outputs]
    label_list = [o["label"] for o in outputs]

    for answer in answers:
        text = answer["answer"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["answer"] = "no"
        else:
            answer["answer"] = "yes"

    for i in range(len(label_list)):
        if label_list[i] == "no":
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer["answer"] == "no":
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))
    data_dict = {
        "acc": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "yes_rate": round(yes_ratio * 100, 2),
    }
    return data_dict


class CaptioningDataset(Dataset):
    def __init__(self, image_root, file_path) -> None:
        super().__init__()
        self.image_root = image_root
        self.data = [json.loads(q) for q in open(file_path, "r")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "image": os.path.join(self.image_root, self.data[index]["image"]),
            "prompt": self.data[index]["text"],
            "label": self.data[index]["label"],
        }


class Collator(VLCollator):
    def __call__(self, batch):
        questions = [b["prompt"] for b in batch]
        images = [b["image"] for b in batch]
        prompt = [self.processor.format_multimodal_prompt(q, img) for q, img in zip(questions, images)]
        inputs = self.processor(texts=prompt, images_path=images, padding_side="left", check_format=False)
        others = [{"label": b["label"]} for b in batch]
        return inputs, others


if __name__ == "__main__":

    args = parse_args()
    image_root = args.image_root
    file_root = args.file_root
    all_results = []
    for root, dirs, files in os.walk(file_root):
        for file in files:
            file_path = os.path.join(root, file)
            if "popular" in file.lower():
                prefix = "popular"
            elif "adversarial" in file.lower():
                prefix = "adv"
            elif "random" in file.lower():
                prefix = "random"
            else:
                raise ValueError(f"Unsupported file: {file}")
            dataset = CaptioningDataset(image_root, file_path)
            # this results is collected from all devices and has a length of len(dataset)
            results = run_vqa(args.model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size)
            if accelerator.is_local_main_process:
                print(f"Results on {file}")
                data_dict = evaluate(results)
                if args.report_to_mysql:
                    try:
                        data_dict = {f"{prefix}_{k}": v for k, v in data_dict.items()}
                        data_dict["tag"] = args.sql_tag
                        log_data_to_mysql(
                            args.sql_host,
                            args.sql_port,
                            args.sql_user,
                            args.sql_password,
                            args.sql_db,
                            args.sql_table,
                            data_dict,
                        )
                    except Exception as e:
                        logger.exception(e)
            results = [r for r in results if r.update(category=file) is None]
            all_results.extend(results)
    if accelerator.is_local_main_process:
        with open(args.output_path, "w") as f:
            json.dump(all_results, f, indent=4)

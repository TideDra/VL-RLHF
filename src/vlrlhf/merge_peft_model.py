from vlrlhf.utils.auto_load import MyAutoModel
from argparse import ArgumentParser
import os
from loguru import logger
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--adapter_path", type=str)
    args = parser.parse_args()
    peft_model = MyAutoModel.from_pretrained(args.adapter_path)
    architectures = peft_model.config.architectures
    logger.info("Merging and unloading model...")
    merged_model = peft_model.merge_and_unload()
    save_path = os.path.join(args.adapter_path, "merged")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path)
    with open(os.path.join(save_path, "config.json"), "r") as f:
        config = json.load(f)
    config["architectures"] = architectures
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    logger.success("Done!")

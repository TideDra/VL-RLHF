import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import gc
import os
import json
import pymysql
from transformers import GenerationConfig


class VLCollator(ABC):
    def __init__(self, processor) -> None:
        super().__init__()
        self.processor = processor

    @abstractmethod
    def __call__(self, batch):
        raise NotImplementedError


model_cache = {}


def load_model_and_processor(model_path, processor_path, **kwargs):
    from ..utils.auto_load import MyAutoModel, MyAutoProcessor
    from peft import PeftModel

    global model_cache
    key = str(model_path) + str(processor_path)
    if key in model_cache:
        return model_cache[key]
    else:
        del model_cache
        gc.collect()
        model_cache = {}
    model = MyAutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)
    if isinstance(model, PeftModel):
        model_path = model.peft_config["default"].base_model_name_or_path

    if processor_path is None:
        processor_path = model_path
    generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = MyAutoProcessor.from_pretrained(processor_path)
    processor.infer()
    generation_kwargs = model.prepare_default_generation_kwargs(generation_config)
    model_cache[key] = (model, processor, generation_kwargs)
    return model, processor, generation_kwargs


def run_vqa(model_path: str, dataset, collator, accelerator, processor_path: str = None, batch_size=16):

    model, processor, generation_kwargs = load_model_and_processor(
        model_path, processor_path, use_flash_attention_2=True
    )
    tokenizer = processor.tokenizer
    model.to(torch.bfloat16)
    model.to(accelerator.device)

    results = []
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collator(processor))
    bar = tqdm(total=len(dataset), disable=not accelerator.is_local_main_process)
    model.eval()
    dataloader = accelerator.prepare(dataloader)

    with torch.inference_mode():
        for inputs, others in dataloader:
            inputs.pop("labels", None)
            outputs = model.generate(**inputs, use_cache=True, **generation_kwargs)
            input_token_len = inputs["input_ids"].shape[1]
            if outputs.shape[1] > input_token_len and torch.all(outputs[:, :input_token_len] == inputs["input_ids"]):
                responses = outputs[:, input_token_len:]
            else:
                responses = outputs
            responses = tokenizer.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            others = accelerator.gather_for_metrics(others)
            responses = accelerator.gather_for_metrics(responses)
            results.extend(
                [other for other, res in zip(others, responses) if other.update(response=res.strip()) is None]
            )
            bar.update(len(responses))
    return results[: len(dataset)]


def run_vqa_ppl(model_path: str, dataset, collator, accelerator, processor_path: str = None, batch_size=16):

    model, processor, generation_kwargs = load_model_and_processor(
        model_path, processor_path, use_flash_attention_2=True
    )
    model.to(torch.bfloat16)
    model.to(accelerator.device)

    results = []
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collator(processor))
    bar = tqdm(total=len(dataset), disable=not accelerator.is_local_main_process)
    model.eval()
    dataloader = accelerator.prepare(dataloader)

    with torch.inference_mode():
        for inputs, others in dataloader:
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            if "labels" in outputs:
                labels = outputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            ).reshape(shift_labels.shape[0], shift_labels.shape[1])
            lens = torch.sum(shift_labels != -100, dim=-1)
            ce_loss = loss.sum(-1) / lens
            others = accelerator.gather_for_metrics(others)
            ce_losses = accelerator.gather(ce_loss)
            ce_losses = ce_losses.cpu().squeeze().tolist()
            results.extend([other for other, res in zip(others, ce_losses) if other.update(ppl=res) is None])
            bar.update(len(ce_losses))
    return results[: len(dataset)]


def run_vqa_sgl(model_path: str, dataset, collator, processor_path: str = None, batch_size=16, endpoint=None):
    from sglang import Runtime, function, image, user, assistant, gen, set_default_backend, RuntimeEndpoint

    @function
    def image_qa(s, image_path, question):
        s += user(image(image_path) + question)
        s += assistant(gen("answer"))

    global model_cache
    if processor_path is None:
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                processor_path = json.load(f).get("_name_or_path", model_path)
        else:
            processor_path = model_path
    if "llava-v1.6-vicuna-7b" in model_path:
        processor_path = "ckpts/llava-1.5-7b-hf"
    if "llava-v1.6-vicuna-13b" in model_path:
        processor_path = "ckpts/llava-1.5-13b-hf"
    if "llava-v1.6-mistral-7b" in model_path:
        processor_path = "ckpts/llava-v1.6-mistral-7b-processor"
    if endpoint is None:
        key = str(model_path) + str(processor_path)
        if key in model_cache:
            runtime = model_cache[key]
        else:
            for v in model_cache.values():
                v.shutdown()
            del model_cache
            gc.collect()
            model_cache = {}
            runtime = Runtime(
                model_path=model_path, tokenizer_path=processor_path, port=30000, tp_size=8, mem_fraction_static=0.7
            )
            model_cache[key] = runtime
    else:
        runtime = RuntimeEndpoint(endpoint)
    set_default_backend(runtime)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    results = []
    all_batches = []
    all_others = []
    for batch, others in dataloader:
        all_batches.extend(batch)
        all_others.extend(others)
    states = image_qa.run_batch(all_batches, max_new_tokens=256, temperature=0, progress_bar=True, num_threads=32)
    results.extend(
        [other for other, res in zip(all_others, states) if other.update(response=res["answer"].strip()) is None]
    )
    return results


def get_model_cache():
    global model_cache
    return model_cache


def log_data_to_mysql(host, port, user, passward, database, table, data_dict):
    db = pymysql.connect(host=host, user=user, passwd=passward, database=database, port=port)
    cursor = db.cursor()
    exp_info = data_dict["tag"]
    exp_info = parse_tag(exp_info)
    tag = exp_info["tag"]
    data_dict["tag"] = tag
    cursor.execute(f"SELECT 1 FROM exps WHERE tag='{tag}' LIMIT 1")
    if not cursor.fetchone():
        keys = []
        values = []
        for k, v in exp_info.items():
            keys.append(k)
            values.append(f"'{v}'" if isinstance(v, str) else str(v))
        keys = ",".join(keys)
        values = ",".join(values)
        sql_code = f"INSERT INTO exps ({keys}) VALUES ({values})"
        cursor.execute(sql_code)
        db.commit()
    values = []
    columns = []
    for k, v in data_dict.items():
        columns.append(k)
        values.append(f"'{v}'" if isinstance(v, str) else str(v))
    # check if tag exist in table
    cursor.execute(f"SELECT 1 FROM {table} WHERE tag='{tag}' LIMIT 1")
    if cursor.fetchone():
        # update
        setting = ",".join([f"{col}={val}" for col, val in zip(columns, values)])
        sql_code = f"UPDATE {table} SET {setting} WHERE tag='{tag}'"
    else:
        # insert
        sql_code = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(values)})"
    cursor.execute(sql_code)
    db.commit()
    db.close()


def parse_tag(tag):
    terms = tag.split(",")
    info = {}
    for term in terms:
        if ":" in term:
            key, value = term.split(":")
        else:
            key, value = term.split("=")
            value = json.loads(value)
        info[key] = value
    return info

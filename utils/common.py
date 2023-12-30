from typing import Literal


def prepare_tokenizer(tokenizer,mode:Literal["train","inference"]):
    if tokenizer.__class__.__name__ == "QWenTokenizer":
        if mode == "train":
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        if mode == "inference":
            pass
    elif tokenizer.__class__.__name__ == "LlamaTokenizer":
        if mode == "train":
            tokenizer.pad_token = tokenizer.unk_token

        if mode == "inference":
            tokenizer.pad_token = tokenizer.bos_token

def get_vision_tower(model):
    if model.__class__.__name__ == "LlavaForRL":
        return model.vision_tower
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return model.transformer.visual
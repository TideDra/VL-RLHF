from typing import Literal, Union
import torch
from transformers import Trainer
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

def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1, padding_side:Literal["right","left"] = "right"
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        if padding_side == "right":
            return torch.cat(
                [
                    tensor,
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )
        elif padding_side == "left":
            return torch.cat(
                [
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor,
                ],
                dim=dim,
            )
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    #! need to be tested
    if trainer.deepspeed:
        torch.cuda.synchronize()

    trainer._save(output_dir)
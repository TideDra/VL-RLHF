from dataclasses import dataclass, field
from typing import Dict, Optional
from dpo_trainer import LLaVADPOTrainer
import torch
from data_utils import make_vlfeedback_paired_dataset, LLaVADPODataCollatorWithPadding
from transformers import HfArgumentParser, TrainingArguments, AutoProcessor, Trainer
from model import LlavaForRL
import os
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    data_dir: Optional[str] = field(default=None,metadata={"help": "path of huggingface dataset cache dir"})
    model_name_or_path: Optional[str] = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the model name"})

    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    wandb_project: Optional[str] = field(default="VL-RLHF", metadata={"help": "wandb project name"})
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args,training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}
    os.environ["WANDB_PROJECT"] = script_args.wandb_project
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # 1. load a pretrained model
    model = LlavaForRL.from_pretrained(script_args.model_name_or_path)
    model.to(compute_dtype)
    model.vision_tower.requires_grad_(False)
    model.config.hidden_size=4096
    model.config.label_pad_token_id=script_args.label_pad_token_id
    model.config.use_cache=False
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = LlavaForRL.from_pretrained(script_args.model_name_or_path)
    model_ref.to(compute_dtype)
    model_ref.config.hidden_size=4096
    model_ref.config.label_pad_token_id=script_args.label_pad_token_id
    model_ref.config.use_cache=False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    processor = AutoProcessor.from_pretrained(script_args.model_name_or_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.unk_token #? unsure which pad_token should be used
        processor.tokenizer.pad_token_id = processor.tokenizer.unk_token_id

    # 4. initialize training arguments:

    local_rank = training_args.local_rank
    dataset = make_vlfeedback_paired_dataset(local_rank,script_args.data_dir)
    dataset_split = dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    data_collator = LLaVADPODataCollatorWithPadding(
        pad_token_id=processor.tokenizer.pad_token_id,
        label_pad_token_id=script_args.label_pad_token_id,
        is_encoder_decoder=model.config.is_encoder_decoder,
        image_processor=processor.image_processor
    )
    # 5. initialize the DPO trainer
    dpo_trainer = LLaVADPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=True,
        label_pad_token_id=script_args.label_pad_token_id,
        data_collator=data_collator,
    )
    dpo_trainer.use_dpo_data_collator = True
    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_state()
    safe_save_model_for_hf_trainer(dpo_trainer,training_args.output_dir)
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
from .utils.data import make_vlfeedback_paired_dataset
from transformers import HfArgumentParser, Trainer
from peft import LoraConfig
import transformers
import os
from .utils.auto_load import MyAutoModel, MyAutoDPOCollator, MyAutoDPOTrainer, MyAutoProcessor
from .utils.common import get_vision_tower, safe_save_model_for_hf_trainer
from transformers import GPTQConfig, deepspeed
from loguru import logger
from transformers.trainer_callback import TrainerCallback

# transformers.logging.set_verbosity_info()
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    score_margin: Optional[float] = field(
        default=-1,
        metadata={
            "help": "Minimal score gap between chosen and rejected responses. Set to -1 to select the pair with the largest score gap for each prompt."
        },
    )
    # training parameters
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "path of huggingface dataset cache dir"}
    )
    model_name_or_path: Optional[str] = field(
        default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the model name"}
    )

    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    max_prompt_length: Optional[int] = field(
        default=128, metadata={"help": "max length of each sample's prompt"}
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "Only used for encoder decoder model. Max target of each sample's prompt"
        },
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"}
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    freeze_vision_tower: bool = field(default=True)
    loss_type: str = field(default='sigmoid')

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = field(default=None)
    lora_bias: str = "none"
    q_lora: bool = False
    bits: int = 4
    modules_to_save: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.lora_target_modules is not None:
            self.lora_target_modules = self.lora_target_modules.split(",")
        if self.modules_to_save is not None:
            self.modules_to_save = self.modules_to_save.split(",")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = False
    project_name: Optional[str] = field(
        default="VL-RLHF", metadata={"help": "wandb project name"}
    )
    group_name: Optional[str] = field(
        default="llava-1.5-7b-dpo", metadata={"help": "wandb group name"}
    )
    resume_from_checkpoint: Optional[bool] = field(default=None)

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, LoraArguments))
    script_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_RUN_GROUP"] = training_args.group_name
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    config = transformers.AutoConfig.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        fp32=True,
    )
    config.use_cache = False

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    model = MyAutoModel.from_pretrained(
        script_args.model_name_or_path,
        config=config,
        device_map=device_map,
        quantization_config=GPTQConfig(bits=lora_args.bits, disable_exllama=True)
        if training_args.use_lora and lora_args.q_lora
        else None,
    )
    model.to(compute_dtype)
    vision_tower = get_vision_tower(model)
    ref_model = None
    if deepspeed.is_deepspeed_zero3_enabled():
        ref_model = MyAutoModel.from_pretrained(
            script_args.model_name_or_path,
            config=config,
            device_map=device_map,
            quantization_config=GPTQConfig(bits=lora_args.bits, disable_exllama=True)
            if training_args.use_lora and lora_args.q_lora
            else None,
        )
        ref_model.to(compute_dtype)
        ref_model.requires_grad_(False)
        ref_model.use_cache = False
        ref_model.config.label_pad_token_id = script_args.label_pad_token_id

    if not training_args.use_lora:
        if script_args.freeze_vision_tower:
            vision_tower.requires_grad_(False)
            if hasattr(vision_tower, "attn_pool"): # follow Qwen-VL default setting
                vision_tower.attn_pool.requires_grad_(True)

    model.config.label_pad_token_id = script_args.label_pad_token_id
    model.config.use_cache = False
    lora_config = None
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias = lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=lora_args.modules_to_save
        )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    processor = MyAutoProcessor.from_pretrained(script_args.model_name_or_path)

    processor.train()

    local_rank = training_args.local_rank
    dataset = make_vlfeedback_paired_dataset(local_rank, script_args.data_dir, script_args.score_margin)
    dataset_split = dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    data_collator = MyAutoDPOCollator(
        script_args.model_name_or_path,
        pad_token_id=processor.tokenizer.pad_token_id,
        label_pad_token_id=script_args.label_pad_token_id,
        is_encoder_decoder=model.config.is_encoder_decoder,
        processor=processor
    )

    dpo_trainer = MyAutoDPOTrainer(
        script_args.model_name_or_path,
        model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        label_pad_token_id=script_args.label_pad_token_id,
        data_collator=data_collator,
        peft_config=lora_config,
        loss_type=script_args.loss_type,
        ref_model=ref_model,
    )
    if training_args.use_lora:
        dpo_trainer.add_callback(PeftSavingCallback())
    dpo_trainer.use_dpo_data_collator = True

    dpo_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    dpo_trainer.save_state()
    safe_save_model_for_hf_trainer(dpo_trainer, training_args.output_dir,lora_args)
    processor.save_pretrained(training_args.output_dir)
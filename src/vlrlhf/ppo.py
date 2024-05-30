from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from vlrlhf.utils.data import build_dataset_from_vlquery_json
from transformers import HfArgumentParser
from peft import LoraConfig
import transformers
import os
from vlrlhf.utils.auto_load import (
    MyAutoProcessor,
    MyAutoModelWithValueHead,
    MyAutoPPOTrainer,
    MyAutoRewardModel,
    MyAutoPPOCollator,
    MyAutoGenerationConfig,
)
from vlrlhf.utils.common import get_vision_tower, safe_save_model_for_ppo_trainer
from transformers import GPTQConfig, deepspeed
from loguru import logger
import trl
from transformers.utils import is_torch_tf32_available
from copy import deepcopy


# transformers.logging.set_verbosity_info()
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the PPO training script.
    """

    # training parameters
    data_dir: Optional[str] = field(default=None, metadata={"help": "path of vlquery dataset"})
    image_root: Optional[str] = field(default=None, metadata={"help": "path of image root"})
    model_name_or_path: Optional[str] = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the model name"})
    reward_adapter: Optional[str] = field(default=None, metadata={"help": "the reward adapter path"})
    reward_adapter_name: Optional[str] = field(default="reward_adapter", metadata={"help": "the reward adapter name"})
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    freeze_vision_tower: bool = field(default=True)
    v_head_init_strategy: Optional[str] = field(default="normal")
    v_head_initializer_range: Optional[float] = field(default=0.2)


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
class PPOConfig(trl.PPOConfig):
    use_lora: bool = False
    use_value_adapter: bool = False
    # TODO: support stand-alone value model
    run_name: Optional[str] = field(default=None)
    project_name: Optional[str] = field(default="VL-RLHF", metadata={"help": "wandb project name"})
    group_name: Optional[str] = field(default="Qwen-VL-Chat-ppo", metadata={"help": "wandb group name"})
    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: Optional[Dict] = field(default_factory=lambda: {"use_reentrant": False})
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    tf32: bool = field(default=False)
    # TODO: support fsdp
    fsdp: str = field(
        default="", metadata={"help": "Fully Sharded Data Parallelism. This feature has not been implemented yet."}
    )
    local_rank: int = field(default=-1, metadata={"help": "local rank for distributed training"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output directory"})
    per_device_gamelog_size: int = field(default=2, metadata={"help": "gamelog size per device"})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "max new tokens for generation"})
    max_length: Optional[int] = field(default=None, metadata={"help": "max length for generation"})

    # TODO: add optimizer and lr_scheduler
    def __post_init__(self):
        super().__post_init__()
        self.tracker_project_name = self.project_name
        self.tracker_kwargs = {"wandb": {"group": self.group_name, "name": self.run_name}}
        if self.fp16 and self.bf16:
            raise ValueError("You can only use one of fp16 and bf16")
        if self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
        if self.per_device_gamelog_size > self.batch_size:
            raise ValueError("per_device_gamelog_size should be less than batch_size")
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.local_rank))
        if self.use_value_adapter and not self.use_lora:
            raise ValueError("You can only use value adapter with a Peft base model. Please set use_lora to True.")
        if self.max_new_tokens is not None and self.max_length is not None:
            raise ValueError("You can only use one of max_new_tokens and max_length")


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, LoraArguments))
    script_args, ppo_config, lora_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.float16 if ppo_config.fp16 else (torch.bfloat16 if ppo_config.bf16 else torch.float32)

    config = transformers.AutoConfig.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
    )

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(ppo_config.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    lora_config = None
    if ppo_config.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=lora_args.modules_to_save,
        )
    if script_args.reward_adapter is not None and script_args.reward_model_name_or_path is not None:
        raise ValueError("You can only use one of reward_adapter and reward_model")
    reward_model = None
    if script_args.reward_model_name_or_path is not None:
        reward_model = MyAutoRewardModel.from_pretrained(script_args.reward_model_name_or_path)
        reward_model.requires_grad_(False)
    model = MyAutoModelWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        config=config,
        device_map=device_map,
        quantization_config=(
            GPTQConfig(bits=lora_args.bits, disable_exllama=True) if ppo_config.use_lora and lora_args.q_lora else None
        ),
        peft_config=lora_config,
        value_adapter_config=deepcopy(lora_config) if ppo_config.use_value_adapter else None,
        reward_adapter=script_args.reward_adapter,
        reward_adapter_name=script_args.reward_adapter_name,
        v_head_init_strategy=script_args.v_head_init_strategy,
        v_head_initializer_range=script_args.v_head_initializer_range,
    )
    model.pretrained_model.config.use_cache = False
    model.to(compute_dtype)
    vision_tower = get_vision_tower(model.pretrained_model)

    if not ppo_config.use_lora:
        if script_args.freeze_vision_tower:
            vision_tower.requires_grad_(False)
            if hasattr(vision_tower, "attn_pool"):  # follow Qwen-VL default setting
                vision_tower.attn_pool.requires_grad_(True)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    processor = MyAutoProcessor.from_pretrained(script_args.model_name_or_path)

    processor.train()

    local_rank = ppo_config.local_rank
    dataset = build_dataset_from_vlquery_json(local_rank, script_args.data_dir, script_args.image_root)
    generation_config = MyAutoGenerationConfig.from_pretrained(script_args.model_name_or_path)
    # ? generation_config may need to be modified for ppo
    generation_config.top_p = 1.0
    if ppo_config.max_new_tokens is not None:
        generation_config.max_new_tokens = ppo_config.max_new_tokens
    if ppo_config.max_length is not None:
        generation_config.max_length = ppo_config.max_length
        generation_config.max_new_tokens = None
    data_collator = MyAutoPPOCollator(script_args.model_name_or_path)
    ppo_trainer = MyAutoPPOTrainer(
        script_args.model_name_or_path,
        config=ppo_config,
        model=model,
        ref_model=None,
        processor=processor,
        dataset=dataset,
        data_collator=data_collator,
        generation_kwargs={"generation_config": generation_config, "use_cache": True},
    )

    ppo_trainer.train()
    safe_save_model_for_ppo_trainer(ppo_trainer, ppo_config.output_dir, lora_args)
    processor.save_pretrained(ppo_config.output_dir)
    ppo_trainer.accelerator.end_training()

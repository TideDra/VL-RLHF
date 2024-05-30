from ..base import (
    VLModelWithValueHead,
    VLRewardModel,
    VLDPODataCollatorWithPadding,
    VLSFTDataCollatorWithPadding,
    VLRMDataCollatorWithPadding,
    VLPPODataCollator,
    VLProcessor,
    VLDPOTrainer,
    VLSFTTrainer,
    VLRMTrainer,
    VLPPOTrainer,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from functools import wraps
from trl import PPOConfig, RewardConfig
from typing import Callable, Dict, List, Literal, Optional, Tuple, Any
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from peft import LoraConfig
from transformers import GPTQConfig, deepspeed
from transformers import (
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
import os
import json
from peft import PeftModel
import torch
from loguru import logger
from importlib import import_module

MODEL_NICKNAME_MAP = {
    "LlavaForConditionalGeneration": "Llava",
    "QWenLMHeadModel": "QwenVL",
    "InstructBlipForConditionalGeneration": "InstructBlip",
    "InstructBlipForRL": "InstructBlip",
    "LlavaNextForConditionalGeneration": "LlavaNext",
    "InternLMXComposer2ForCausalLM": "InternLMXC2",
}
FLASH_ATTN_MODELS = [
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextForRL",
    "LlavaForRL",
    "InternLMXC2",
    "InternLMXComposer2ForCausalLM",
]


def auto_core_mapper(architecture):
    module_name = f".{MODEL_NICKNAME_MAP[architecture]}"
    return import_module(module_name, package="vlrlhf.models").core_mapper


class MyAutoModel:
    @staticmethod
    @wraps(AutoModelForCausalLM.from_pretrained)
    def from_pretrained(model_name_or_path, *model_args, **kwargs) -> PreTrainedModel:
        adapter_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            with open(adapter_path, "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config["base_model_name_or_path"]
            config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
            architecture = config.architectures[0]
            if kwargs.get("use_flash_attention_2", False):
                if architecture not in FLASH_ATTN_MODELS:
                    logger.warning(f"{architecture} does not support flash attention 2. Disabling.")
                    kwargs["use_flash_attention_2"] = False
            model = auto_core_mapper(architecture).model.from_pretrained(
                base_model_name, trust_remote_code=True, *model_args, **kwargs
            )
            peft_model = PeftModel.from_pretrained(model, model_name_or_path)
            return peft_model
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            architecture = config.architectures[0]
            if kwargs.get("use_flash_attention_2", False):
                if architecture not in FLASH_ATTN_MODELS:
                    logger.warning(f"{architecture} does not support flash attention 2. Disabling.")
                    kwargs["use_flash_attention_2"] = False
            return auto_core_mapper(architecture).model.from_pretrained(
                model_name_or_path, trust_remote_code=True, *model_args, **kwargs
            )


class MyAutoRewardModel:
    @staticmethod
    @wraps(AutoModelForCausalLM.from_pretrained)
    def from_pretrained(model_name_or_path, *model_args, **kwargs) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        return auto_core_mapper(architecture).reward_model.from_pretrained(
            model_name_or_path, trust_remote_code=True, *model_args, **kwargs
        )


class MyAutoModelWithValueHead:
    @staticmethod
    @wraps(VLModelWithValueHead.from_pretrained)
    def from_pretrained(model_name_or_path, *model_args, **kwargs) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        return auto_core_mapper(architecture).value_model.from_pretrained(
            model_name_or_path, trust_remote_code=True, *model_args, **kwargs
        )


class MyAutoDPOCollator(VLDPODataCollatorWithPadding):
    def __new__(
        cls,
        model_name_or_path: str,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: Optional[bool] = False,
        processor: Optional[Any] = None,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        collator = auto_core_mapper(architecture).dpo_collator
        return collator(pad_token_id, label_pad_token_id, is_encoder_decoder, processor)

    def __init__(
        self,
        model_name_or_path: str,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: Optional[bool] = False,
        processor: Optional[Any] = None,
    ): ...


class MyAutoSFTCollator(VLSFTDataCollatorWithPadding):
    def __new__(
        cls, model_name_or_path: str, pad_token_id: int, label_pad_token_id: int, processor: Optional[Any] = None
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        collator = auto_core_mapper(architecture).sft_collator
        return collator(pad_token_id, label_pad_token_id, processor)

    def __init__(
        self, model_name_or_path: str, pad_token_id: int, label_pad_token_id: int, processor: Optional[Any] = None
    ): ...


class MyAutoRMCollator(VLRMDataCollatorWithPadding):
    def __new__(
        cls,
        model_name_or_path: str,
        pad_token_id: int = 0,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        collator = auto_core_mapper(architecture).reward_collator
        return collator(pad_token_id)

    def __init__(
        self,
        model_name_or_path: str,
        pad_token_id: int = 0,
    ): ...


class MyAutoPPOCollator(VLPPODataCollator):
    def __new__(
        cls,
        model_name_or_path: str,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        collator = auto_core_mapper(architecture).ppo_collator
        return collator()

    def __init__(
        self,
        model_name_or_path: str,
    ): ...


class MyAutoDPOTrainer(VLDPOTrainer):
    def __new__(
        cls,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str = None,
        ref_model: PreTrainedModel | Module | str | None = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "ddpo"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        max_prompt_length: int | None = None,
        max_target_length: int | None = None,
        peft_config: Dict | None = None,
        is_encoder_decoder: bool | None = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Callable[[EvalLoopOutput], Dict] | None = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        trainer = auto_core_mapper(architecture).dpo_trainer
        return trainer(
            model,
            ref_model,
            beta,
            label_smoothing,
            loss_type,
            args,
            data_collator,
            label_pad_token_id,
            padding_value,
            truncation_mode,
            train_dataset,
            eval_dataset,
            processor,
            model_init,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            max_length,
            max_prompt_length,
            max_target_length,
            peft_config,
            is_encoder_decoder,
            disable_dropout,
            generate_during_eval,
            compute_metrics,
            precompute_ref_log_probs,
            dataset_num_proc,
            model_init_kwargs,
            ref_model_init_kwargs,
            model_adapter_name,
            ref_adapter_name,
            reference_free,
        )

    def __init__(
        self,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str = None,
        ref_model: PreTrainedModel | Module | str | None = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto", "ddpo"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = ...,
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        max_prompt_length: int | None = None,
        max_target_length: int | None = None,
        peft_config: Dict | None = None,
        is_encoder_decoder: bool | None = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Callable[[EvalLoopOutput], Dict] | None = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
    ): ...


class MyAutoProcessor:
    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs) -> VLProcessor:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        return auto_core_mapper(architecture).processor(model_name_or_path, trust_remote_code=True, **kwargs)


class MyAutoSFTTrainer(VLSFTTrainer):
    def __new__(
        cls,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str | None = None,
        args: TrainingArguments | None = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer | LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        peft_config: Dict | None = None,
        dataset_text_field: str | None = None,
        packing: bool | None = True,
        formatting_func: Callable[..., Any] | None = None,
        max_seq_length: int | None = None,
        infinite: bool | None = None,
        num_of_sequences: int | None = 1024,
        chars_per_token: float | None = 3.6,
        dataset_num_proc: int | None = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: float | None = None,
        model_init_kwargs: Dict | None = None,
        dataset_kwargs: Dict | None = None,
        eval_packing: bool | None = None,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        trainer = auto_core_mapper(architecture).sft_trainer
        return trainer(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processor,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            peft_config,
            dataset_text_field,
            packing,
            formatting_func,
            max_seq_length,
            infinite,
            num_of_sequences,
            chars_per_token,
            dataset_num_proc,
            dataset_batch_size,
            neftune_noise_alpha,
            model_init_kwargs,
            dataset_kwargs,
            eval_packing,
        )

    def __init__(
        self,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str | None = None,
        args: TrainingArguments | None = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer | LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        peft_config: Dict | None = None,
        dataset_text_field: str | None = None,
        packing: bool | None = False,
        formatting_func: Callable[..., Any] | None = None,
        max_seq_length: int | None = None,
        infinite: bool | None = None,
        num_of_sequences: int | None = 1024,
        chars_per_token: float | None = 3.6,
        dataset_num_proc: int | None = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: float | None = None,
        model_init_kwargs: Dict | None = None,
        dataset_kwargs: Dict | None = None,
        eval_packing: bool | None = None,
    ): ...


class MyAutoRMTrainer(VLRMTrainer):
    def __new__(
        cls,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module = None,
        args: RewardConfig | None = None,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        peft_config: Dict | None = None,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]
        trainer = auto_core_mapper(architecture).reward_trainer
        return trainer(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processor,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            max_length,
            peft_config,
        )

    def __init__(
        self,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module = None,
        args: RewardConfig | None = None,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        peft_config: Dict | None = None,
    ): ...


class MyAutoPPOTrainer(VLPPOTrainer):
    def __new__(
        cls,
        model_name_or_path: str = None,
        config: PPOConfig = None,
        model: VLModelWithValueHead = None,
        ref_model: VLModelWithValueHead | None = None,
        reward_model: VLRewardModel | None = None,
        processor: VLProcessor | None = None,
        dataset: Any | Dataset | None = None,
        optimizer: Optimizer | None = None,
        data_collator: Optional[Callable] = None,
        num_shared_layers: int | None = None,
        lr_scheduler: _LRScheduler | None = None,
        generation_kwargs: dict = ...,
    ):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = model_config.architectures[0]
        trainer = auto_core_mapper(architecture).ppo_trainer
        return trainer(
            config,
            model,
            ref_model,
            reward_model,
            processor,
            dataset,
            optimizer,
            data_collator,
            num_shared_layers,
            lr_scheduler,
            generation_kwargs,
        )

    def __init__(
        self,
        model_name_or_path: str = None,
        config: PPOConfig = None,
        model: VLModelWithValueHead = None,
        ref_model: VLModelWithValueHead | None = None,
        reward_model: VLRewardModel | None = None,
        processor: VLProcessor | None = None,
        dataset: Any | Dataset | None = None,
        optimizer: Optimizer | None = None,
        data_collator: Optional[Callable] = None,
        num_shared_layers: int | None = None,
        lr_scheduler: _LRScheduler | None = None,
        generation_kwargs: dict = ...,
    ): ...


def auto_load_rlmodel(script_args, training_args, lora_args):
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    config = AutoConfig.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = False
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning("FSDP or ZeRO3 are not compatible with QLoRA.")

    model = MyAutoModel.from_pretrained(
        script_args.model_name_or_path,
        config=config,
        device_map=device_map,
        quantization_config=(
            GPTQConfig(bits=lora_args.bits, disable_exllama=True)
            if training_args.use_lora and lora_args.q_lora
            else None
        ),
        use_flash_attention_2=training_args.use_flash_attention_2,
        torch_dtype=compute_dtype,
    )
    ref_model = None
    """
    if deepspeed.is_deepspeed_zero3_enabled():
        ref_model = MyAutoModel.from_pretrained(
            script_args.model_name_or_path,
            config=config,
            device_map=device_map,
            quantization_config=GPTQConfig(bits=lora_args.bits, disable_exllama=True)
            if training_args.use_lora and lora_args.q_lora
            else None,
            use_flash_attention_2=training_args.use_flash_attention_2
        )
        ref_model.to(compute_dtype)
        ref_model.requires_grad_(False)
        ref_model.use_cache = False
        ref_model.config.label_pad_token_id = script_args.label_pad_token_id
    """
    if script_args.freeze_vision_tower:
        model.freeze_vision_tower()

    model.config.label_pad_token_id = script_args.label_pad_token_id
    model.config.use_cache = False
    lora_config = None
    if training_args.use_lora:
        if lora_args.lora_target_modules == "auto":
            lora_args.lora_target_modules = model.default_lora_target
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=lora_args.modules_to_save,
        )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    return model, ref_model, lora_config

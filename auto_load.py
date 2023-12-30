from components.model import LlavaForRL
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from components.collator import LLaVADPODataCollatorWithPadding, QwenVLDPODataCollatorWithPadding, VLDPODataCollatorWithPadding
from components.processor import LlavaProcessor, QwenVLProcessor, VLProcessor
from components.dpo_trainer import LLaVADPOTrainer, QwenVLDPOTrainer
from typing import Optional, Any
from functools import wraps
from trl.trainer import DPOTrainer
from typing import Callable, Dict, List, Literal, Optional, Tuple,Any
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from transformers.generation import GenerationConfig
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
AUTOMODEL_MAP = {
    "LlavaForConditionalGeneration": LlavaForRL,
    "QWenLMHeadModel": AutoModelForCausalLM
}

AUTOCOLLATOR_MAP = {
    "LlavaForConditionalGeneration": LLaVADPODataCollatorWithPadding,
    "QWenLMHeadModel": QwenVLDPODataCollatorWithPadding
}

AUTOTRAINER_MAP = {
    "LlavaForConditionalGeneration": LLaVADPOTrainer,
    "QWenLMHeadModel": QwenVLDPOTrainer
}

AUTOPROCESSOR_MAP = {
    "LlavaForConditionalGeneration": LlavaProcessor,
    "QWenLMHeadModel": QwenVLProcessor
}


class MyAutoModel:
    @staticmethod
    @wraps(AutoModelForCausalLM.from_pretrained)
    def from_pretrained(model_name_or_path, *model_args, **kwargs) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(model_name_or_path)
        architecture = config.architectures[0]
        return AUTOMODEL_MAP[architecture].from_pretrained(model_name_or_path, *model_args, **kwargs)

class MyAutoCollator(VLDPODataCollatorWithPadding):
    def __new__(cls,model_name_or_path:str,  pad_token_id: int = 0,label_pad_token_id: int = -100,is_encoder_decoder: Optional[bool] = False,processor: Optional[Any] = None):
        config = AutoConfig.from_pretrained(model_name_or_path,trust_remote_code=True)
        architecture = config.architectures[0]
        collator = AUTOCOLLATOR_MAP[architecture]
        return collator(pad_token_id, label_pad_token_id, is_encoder_decoder, processor)
    def __init__(self,model_name_or_path:str,  pad_token_id: int = 0,label_pad_token_id: int = -100,is_encoder_decoder: Optional[bool] = False,processor: Optional[Any] = None):
        pass

class MyAutoDPOTrainer(DPOTrainer):
    def __new__(cls,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str = None,
        ref_model: PreTrainedModel | Module | str | None = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None, #* replace tokenizer with processor. processor is a class that contains tokenizer and image processor
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
        model_init_kwargs: Dict | None = None,
        ref_model_init_kwargs: Dict | None = None,
        ):
        config = AutoConfig.from_pretrained(model_name_or_path,trust_remote_code=True)
        architecture = config.architectures[0]
        trainer = AUTOTRAINER_MAP[architecture]
        return trainer(model, ref_model, beta, label_smoothing, loss_type, args, data_collator, label_pad_token_id, padding_value, truncation_mode, train_dataset, eval_dataset, processor, model_init, callbacks, optimizers, preprocess_logits_for_metrics, max_length, max_prompt_length, max_target_length, peft_config, is_encoder_decoder, disable_dropout, generate_during_eval, compute_metrics, precompute_ref_log_probs, model_init_kwargs, ref_model_init_kwargs)

    def __init__(self,
        model_name_or_path: str = None,
        model: PreTrainedModel | Module | str = None,
        ref_model: PreTrainedModel | Module | str | None = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None, #* replace tokenizer with processor. processor is a class that contains tokenizer and image processor
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
        model_init_kwargs: Dict | None = None,
        ref_model_init_kwargs: Dict | None = None,):
        pass

class MyAutoProcessor:
    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs) -> VLProcessor:
        config = AutoConfig.from_pretrained(model_name_or_path)
        architecture = config.architectures[0]
        return AUTOPROCESSOR_MAP[architecture](model_name_or_path, **kwargs)

class MyAutoGenerationKwargs:
    @staticmethod
    def from_pretrained(model_name_or_path) -> Dict[str, Any]:
        config = AutoConfig.from_pretrained(model_name_or_path)
        architecture = config.architectures[0]
        generation_config = GenerationConfig.from_pretrained(model_name_or_path,trust_remote_code=True)
        if architecture == "LlavaForConditionalGeneration":
            generation_config.pad_token_id = generation_config.bos_token_id
            generation_config.temperature = 0.2
            generation_config.max_new_tokens = 1024

            
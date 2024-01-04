from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import RewardTrainer
from trl.trainer.training_configs import RewardConfig
from components.processor import VLProcessor
import torch
from components.collator import VLRMDataCollatorWithPadding
from abc import ABC
class VLRMTrainer(RewardTrainer,ABC):
    def __init__(
        self,
        model: PreTrainedModel | Module = None,
        args: RewardConfig | None = None,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None,None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        peft_config: Dict | None = None,
    ):
        self.processor = processor
        self.max_length = max_length
        train_dataset = train_dataset.map(
            self.tokenize_row,
            remove_columns=train_dataset.column_names,
            keep_in_memory=True
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_row,
                remove_columns=eval_dataset.column_names,
                keep_in_memory=True
            )

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processor.tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            max_length,
            peft_config,
        )

    def tokenize_row(self,element):
        prompt = element["prompt"]
        chosen = element["chosen"]
        rejected = element["rejected"]
        prompt = self.processor.format_multimodal_prompt(prompt, element["img_path"])
        # format for preprocessing
        chosen_conv = self.processor.make_single_turn_conv(prompt, chosen)
        rejected_conv = self.processor.make_single_turn_conv(prompt, rejected)
        # preprocess using Qwen-VL's own method
        # note that labels are already set here
        processed_chosen_conv = self.processor.process_batch_conv([chosen_conv])
        chosen_full_tokens = processed_chosen_conv["full"]
        processed_rejected_conv = self.processor.process_batch_conv([rejected_conv])
        rejected_full_tokens = processed_rejected_conv["full"]
        
        return {
            "input_ids_chosen": chosen_full_tokens["input_ids"][0][:self.max_length],
            "attention_mask_chosen": chosen_full_tokens["attention_mask"][0][:self.max_length],
            "input_ids_rejected": rejected_full_tokens["input_ids"][0][:self.max_length],
            "attention_mask_rejected": rejected_full_tokens["attention_mask"][0][:self.max_length],
        }

class LlavaRMTrainer(VLRMTrainer):
    ...

class QwenVLRMTrainer(VLRMTrainer):
    ...
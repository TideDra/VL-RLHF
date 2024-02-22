from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from datasets import Dataset
from peft import PeftConfig
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from .processor import VLProcessor
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers import Trainer
from abc import ABC
class VLSFTTrainer(Trainer, ABC):
    def __init__(
        self,
        model: PreTrainedModel | Module = None,
        args: TrainingArguments = None,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        processor: VLProcessor | None = None,
        max_seq_length: int | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None,None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
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
            self.processor.tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def tokenize_row(self,element):
        conv = element["conversations"]
        image = element["image"]
        first_sentence = conv[0]["value"]
        first_sentence = self.processor.format_multimodal_prompt(
            first_sentence, image
        )
        conv[0]["value"] = first_sentence
        tokens = self.processor.process_batch_conv([conv])["full"]
        tokens = {
            k: v[0][:self.max_seq_length]
            for k, v in tokens.items()
        }
        return tokens

class QwenVLSFTTrainer(VLSFTTrainer):
    ...

class LlavaSFTTRainer(VLSFTTrainer):
    ...
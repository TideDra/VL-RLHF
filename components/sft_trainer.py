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

from components.processor import VLProcessor
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers import Trainer
from abc import ABC
import torch
from components.collator import VLSFTDataCollatorWithPadding
from transformers.trainer_pt_utils import LabelSmoother
class VLSFTTrainer(Trainer, ABC):
    def __init__(
        self,
        model: PreTrainedModel | Module = None,
        args: TrainingArguments = None,
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
        train_dataset = self._prepare_dataset(train_dataset, max_seq_length)
        data_collator = VLSFTDataCollatorWithPadding(self.processor.tokenizer.pad_token_id,LabelSmoother.ignore_index)
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

    def _prepare_dataset(
        self,
        dataset,
        max_seq_length,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # check if torch dataset / dataloader and do nothing
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset)
        ):
            return dataset

        def tokenize_row(element):
            conv = element["conversations"]
            image = element["image"]
            first_sentence = conv[0]["value"]
            first_sentence = self.processor.format_multimodal_prompt(
                first_sentence, image
            )
            conv[0]["value"] = first_sentence
            tokens = self.processor.process_batch_conv([conv])["full"]
            tokens = {
                k: v[0][:max_seq_length]
                for k, v in tokens.items()
            }
            return tokens

        tokenized_dataset = dataset.map(
            tokenize_row,
            remove_columns=dataset.column_names,
            keep_in_memory=True
        )

        return tokenized_dataset

class QwenVLSFTTrainer(VLSFTTrainer):
    ...

class LlavaSFTTRainer(VLSFTTrainer):
    ...
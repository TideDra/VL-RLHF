from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
import torch
from torch.utils.data import DataLoader
import random
import wandb

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value *
                torch.ones(*pad_size, dtype=tensor.dtype,
                           device=tensor.device),
            ],
            dim=dim,
        )


class LLaVADPOTrainer(DPOTrainer):
    def __init__(
        self,
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
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[
            Tensor, Tensor], Tensor] | None = None,
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
        super().__init__(
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
            tokenizer,
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
            model_init_kwargs,
            ref_model_init_kwargs,
        )

    def tokenize_row(self, feature, model: Union[PreTrainedModel, Module] = None) -> Dict:
        batch = super().tokenize_row(feature, model)
        batch['img_path'] = feature['img_path']
        return batch

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = super().concatenated_inputs(
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device)
        # duplicate image in batchsize dimension
        concatenated_img = batch['pixel_values'].repeat(2, 1, 1, 1)
        concatenated_batch['pixel_values'] = concatenated_img
        return concatenated_batch

    def concatenated_forward(
        self, model: Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        output = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            pixel_values=concatenated_batch["pixel_values"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            logp_labels=concatenated_batch["concatenated_labels"],
            **model_kwargs,
        )
        all_logits = output.logits
        concatenated_labels_after_merging = output.logp_labels
        #* concatenated_batch["concatenated_labels"] is not used here because model.forward merge image features into input_embeds, so the final sequenlength is 
        #* longer than the original labels. we set the correct labels to output.logp_labels
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_labels_after_merging,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # if reference_output in batch use that otherwise use the reference model
        if "reference_output" in batch:
            reference_output = batch["reference_output"]
        else:
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_output = self.model.generate(
                        input_ids=batch["prompt_input_ids"],
                        pixel_values=batch["pixel_values"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                reference_output = self.ref_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(
            reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

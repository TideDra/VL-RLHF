from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
import torch
from .processor import VLProcessor
from abc import ABC
from utils.common import pad_to_length
from peft import PeftModel
from transformers.trainer import unwrap_model
import os

class VLDPOTrainer(DPOTrainer, ABC):
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
        processor: VLProcessor | None = None, #* replace tokenizer with processor. processor is a class that contains tokenizer and image processor
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
        model_init_kwargs: Dict | None = None,
        ref_model_init_kwargs: Dict | None = None,
    ):
        self.processor = processor
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
            processor.tokenizer,
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
    
    def tokenize_row(self, feature, model: PreTrainedModel | Module = None) -> Dict:
        #* tokenize_row manages all processing steps for tokenizing all texts to input_ids, labels, attention_mask, etc.
        prompt = self.processor.format_multimodal_prompt(feature['prompt'],feature['img_path']) # add image placeholder to prompt
        prompt = self.processor.make_single_turn_conv(prompt,"") # This returns [{"user":<image>\n<prompt>,"assistant":""}]
        prompt = self.processor.process_batch_conv(prompt,system_message=None) # This returns {"prompt":None,"answer":None,"full":None,"raw_str":[...]}
        feature['prompt'] = prompt['raw_str'][0] # This returns "USER: <image>\n<prompt> ASSISTANT:"
        batch = super().tokenize_row(feature, model)
        batch['img_path'] = feature['img_path']
        return batch

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        #* merge peft model before saving
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        supported_classes = (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = self.model
        if not isinstance(model, supported_classes): # model is wrapped
            model = unwrap_model(model)
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            if state_dict is None:
                state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            model.save_pretrained(
                    output_dir, state_dict=cpu_state_dict, safe_serialization=self.args.save_safetensors
                )

        else: # model is not wrapped
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            if state_dict is None:
                state_dict = model.state_dict()
            model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class LLaVADPOTrainer(VLDPOTrainer):
    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = super().concatenated_inputs(
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device
        )
        # duplicate image in batchsize dimension
        concatenated_img = batch["pixel_values"].repeat(2, 1, 1, 1)
        concatenated_batch["pixel_values"] = concatenated_img
        return concatenated_batch

    def concatenated_forward(
        self, model: Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
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
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
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
        # * concatenated_batch["concatenated_labels"] is not used here because model.forward merge image features into input_embeds, so the final sequenlength is
        # * longer than the original labels. we set the correct labels to output.logp_labels
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

    def get_batch_samples(
        self, model, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
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
            policy_output, self.max_length, self.tokenizer.pad_token_id
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output = pad_to_length(
            reference_output, self.max_length, self.tokenizer.pad_token_id
        )
        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output, skip_special_tokens=True
        )

        return policy_output_decoded, reference_output_decoded


class QwenVLDPOTrainer(VLDPOTrainer):
    def tokenize_row(self, feature, model: PreTrainedModel | Module = None) -> Dict:
        #FIXME: try to use raw tokenize_raw code
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        prompt = self.processor.format_multimodal_prompt(prompt, feature["img_path"])
        # format for preprocessing

        chosen_conv = self.processor.make_single_turn_conv(prompt, chosen)
        rejected_conv = self.processor.make_single_turn_conv(prompt, rejected)

        # preprocess using Qwen-VL's own method
        # note that labels are already set here
        processed_chosen_conv = self.processor.process_batch_conv([chosen_conv])
        prompt_tokens = processed_chosen_conv["prompt"]
        chosen_tokens = processed_chosen_conv["answer"]
        processed_rejected_conv = self.processor.process_batch_conv([rejected_conv])
        rejected_tokens = processed_rejected_conv["answer"]
        prompt_tokens = {k: v[0] for k, v in prompt_tokens.items()}
        chosen_tokens = {k: v[0] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[0] for k, v in rejected_tokens.items()}

        eos_token_id = self.tokenizer.eos_token_id
        # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
        eos_indices_prompt = [
            i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
        ]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p
            for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [
            i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
        ]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p
            for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [
            i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id
        ]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p
            for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["labels"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["labels"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {
                    k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
                }
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {
                    k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()
                }
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in chosen_tokens.items()
            }
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length]
                for k, v in rejected_tokens.items()
            }

        # Create labels
        chosen_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_tokens = {
            k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
        }
        chosen_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])
        rejected_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(prompt_tokens["input_ids"])
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        for k, toks in {
            "chosen_": chosen_tokens,
            "rejected_": rejected_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        return batch

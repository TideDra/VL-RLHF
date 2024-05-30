from typing import Callable, Dict, List, Literal, Optional, Tuple, Any, Union
from datasets import Dataset
from peft import PeftConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from trl import DPOTrainer, PPOTrainer, PPOConfig, SFTTrainer
from .processor import VLProcessor
from .model import VLModelWithValueHead, VLRewardModel
import wandb
from loguru import logger
from accelerate.utils import gather_object
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from trl import RewardTrainer
from trl.trainer.reward_config import RewardConfig
from accelerate.utils import tqdm
from abc import ABC
from contextlib import nullcontext
from ..utils.common import pad_to_length
from ..utils.diff_lib import get_diff_ids
import gc


class VLDPOTrainer(DPOTrainer, ABC):
    def __init__(
        self,
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
            dataset_num_proc,
            model_init_kwargs,
            ref_model_init_kwargs,
            model_adapter_name,
            ref_adapter_name,
            reference_free,
        )

    def tokenize_row(self, feature, model: PreTrainedModel | Module = None) -> Dict:
        prompt = self.processor.format_multimodal_prompt(
            feature["prompt"], feature["img_path"]
        )  # add image placeholder to prompt
        prompt = self.processor.make_single_turn_conv(
            prompt, ""
        )  # This returns [{"user":<image>\n<prompt>,"assistant":""}]
        prompt = self.processor.process_batch_conv(
            [prompt], system_message=None, add_end_for_empty_value=False
        )  # This returns {"prompt":None,"answer":None,"full":None,"raw_str":[...]}
        prompt_raw_str = prompt["raw_str"][0]  # This returns "USER: <image>\n<prompt> ASSISTANT:"
        assistant_end = self.processor.chat_template.assistant_end
        feature["chosen"] += assistant_end
        feature["rejected"] += assistant_end
        feature["prompt"] = prompt_raw_str
        batch = super().tokenize_row(feature, model)
        batch["img_path"] = feature["img_path"]
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
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device
        )
        if "img_input_dict" in batch:
            # duplicate img_input in batchsize dimension
            concatenated_img_input_dict = {}
            for k, v in batch["img_input_dict"].items():
                if isinstance(v, torch.Tensor):
                    concatenated_img_input_dict[k] = torch.cat([v, v], dim=0)
                elif isinstance(v, list):
                    concatenated_img_input_dict[k] = v + v
                else:
                    raise ValueError(f"Unsupported type {type(v)} for concatenation.")
            concatenated_batch["concatenated_img_input_dict"] = concatenated_img_input_dict
        return concatenated_batch

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        mask_shared_tokens: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if mask_shared_tokens:
            len_chosen = labels.shape[0] // 2
            assert len_chosen * 2 == labels.shape[0]
            chosen_labels = labels[:len_chosen]
            rejected_labels = labels[len_chosen:]
            chosen_shared_mask = torch.full_like(chosen_labels, False, dtype=torch.bool)
            rejected_shared_mask = torch.full_like(rejected_labels, False, dtype=torch.bool)
            min_match_size = 3
            for idx, (chosen_label, rejected_label) in enumerate(zip(chosen_labels, rejected_labels)):
                c_mod, r_mod = get_diff_ids(
                    chosen_label.tolist(), rejected_label.tolist(), min_match_size=min_match_size
                )
                chosen_shared_mask[idx][c_mod] = True
                rejected_shared_mask[idx][r_mod] = True
            shared_mask = torch.cat([chosen_shared_mask, rejected_shared_mask], dim=0)
            loss_mask &= shared_mask
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

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
            else {"use_cache": False}
        )
        if "concatenated_img_input_dict" in concatenated_batch:
            model_kwargs.update(concatenated_batch["concatenated_img_input_dict"])
        output = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            labels=concatenated_batch["concatenated_labels"],
            **model_kwargs,
        )
        all_logits = output.logits
        if hasattr(output, "labels"):
            final_labels = output.labels
        else:
            final_labels = concatenated_batch["concatenated_labels"]
        all_logps = self.get_batch_logps(
            all_logits,
            final_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            mask_shared_tokens=self.loss_type == "ddpo",
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid" or self.loss_type == "ddpo":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()  # noqa
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)
            ).detach()  # noqa
        )

        return losses, chosen_rewards, rejected_rewards

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # fix memory increasing issue: https://github.com/huggingface/trl/issues/1377
        loss_step = super().training_step(model, inputs)
        torch.cuda.empty_cache()
        gc.collect()
        return loss_step

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explictly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        others = {}
        if "img_input_dict" in batch:
            others.update(batch["img_input_dict"])
        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **others,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            **others,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **others,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded


class VLPPOTrainer(PPOTrainer, ABC):
    def __init__(
        self,
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
        generation_kwargs: dict = {},
    ):
        if getattr(config, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model.gradient_checkpointing_enable(config.gradient_checkpointing_kwargs)
        self.processor = processor
        self.generation_kwargs = generation_kwargs
        if model.supports_rm_adapter and reward_model is not None:
            raise ValueError(
                """Model supports reward model adapter, but you also give a reward model.
                Please only use one of them to compute rewards."""
            )
        self.reward_model = reward_model
        dataset = dataset.map(self.tokenize_row)
        super().__init__(
            config,
            model,
            ref_model,
            processor.tokenizer,
            dataset,
            optimizer,
            data_collator,
            num_shared_layers,
            lr_scheduler,
        )
        self.is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )
        if self.is_deepspeed_used:
            # without this, the program will stuck at pad_across_processes in zero2/3
            self.generation_kwargs["synced_gpus"] = True

    def tokenize_row(self, element):
        query = element["query"]
        image = element["image"]
        query = self.processor.format_multimodal_prompt(query, image)
        conv = self.processor.make_single_turn_conv(query, "")

        tokens = self.processor.process_batch_conv([conv])["full"]
        tokens = {k: v[0] for k, v in tokens.items() if k != "labels"}  # we don't need labels
        tokens["query"] = query
        tokens["image"] = image
        return tokens

    def train(self):
        if self.accelerator.is_main_process:
            bar = tqdm(total=len(self.dataloader), desc="Launching...")
        for epoch, batch in enumerate(self.dataloader):
            query_tensors = batch["input_ids"]
            # Get response from SFTModel
            if self.accelerator.is_main_process:
                bar.set_description("generating response")
            with torch.no_grad():
                response_tensors = self.generate(
                    query_tensors,
                    return_prompt=False,
                    batch_size=32,
                    **self.generation_kwargs,
                )

            batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            # Compute reward score
            convs = [self.processor.make_single_turn_conv(q, r) for q, r in zip(batch["query"], batch["response"])]
            inputs = self.processor.process_batch_conv(convs)["full"]
            inputs.pop("labels")
            padding_side_default = self.tokenizer.padding_side
            if not self.is_encoder_decoder:
                self.tokenizer.padding_side = "left"
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                return_tensors="pt",
            ).to(self.current_device)
            self.tokenizer.padding_side = padding_side_default

            if self.accelerator.is_main_process:
                bar.set_description_str("computing reward score")
            skip_batch = False
            try:
                if self.reward_model is not None:
                    with torch.no_grad():
                        rewards = self.reward_model(**padded_inputs)[0]
                else:
                    rewards = self.accelerator.unwrap_model(self.model).compute_reward_score(
                        **padded_inputs
                    )  # compute_reward_score sets no_grad
            except Exception as e:
                logger.warning(
                    "Error when computing reward score. Skip this batch. See the following exception for more details."
                )
                logger.exception(e)
                print(batch["response"])
                skip_batch = True
            skip_batch = [skip_batch]
            gather_object(skip_batch)
            if torch.tensor(skip_batch).any():
                print(skip_batch)
                continue
            # self.step needs a list of rewards, then it turn the list into a tensor again. This is really stupid.
            rewards = [reward for reward in rewards]
            # Run PPO step
            # we don't pass response_mask because self.generate has already removed padding
            if self.accelerator.is_main_process:
                bar.set_description_str("running ppo step")
            stats = self.step(query_tensors, response_tensors, rewards)
            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.current_device
            )  # bf16 cannot be turned to numpy for logging
            batch = {k: v[: self.config.per_device_gamelog_size] for k, v in batch.items()}
            rewards = rewards[: self.config.per_device_gamelog_size]
            batch["image"] = [wandb.Image(image) for image in batch["image"]]
            batch["query"] = [self.processor.remove_image_placeholder(query) for query in batch["query"]]
            self.log_stats(stats, batch, rewards, columns_to_log=["image", "query", "response"])
            if self.accelerator.is_main_process:
                bar.update()


class VLRMTrainer(RewardTrainer, ABC):
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
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_length: int | None = None,
        peft_config: Dict | None = None,
    ):
        self.processor = processor
        self.max_length = max_length
        train_dataset = train_dataset.map(
            self.tokenize_row,
            remove_columns=train_dataset.column_names,
            # keep_in_memory=True
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_row,
                remove_columns=eval_dataset.column_names,
                # keep_in_memory=True
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
            None,
            peft_config,
        )
        self.use_reward_data_collator = True  # silence warning

    def tokenize_row(self, element):
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
            "input_ids_chosen": chosen_full_tokens["input_ids"][0][: self.max_length],
            "attention_mask_chosen": chosen_full_tokens["attention_mask"][0][: self.max_length],
            "input_ids_rejected": rejected_full_tokens["input_ids"][0][: self.max_length],
            "attention_mask_rejected": rejected_full_tokens["attention_mask"][0][: self.max_length],
        }


class VLSFTTrainer(SFTTrainer, ABC):
    def __init__(
        self,
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
        peft_config: PeftConfig | None = None,
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
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.tokenizer = processor.tokenizer
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

    def tokenize_row(self, element):
        conv = element["conversations"]
        image = element["img_path"]
        first_sentence = conv[0]["value"]
        first_sentence = self.processor.format_multimodal_prompt(first_sentence, image)
        conv[0]["value"] = first_sentence
        tokens = self.processor.process_batch_conv([conv])["full"]  # process_batch_conv adds bos token but no eos token
        tokens = {k: v[0] for k, v in tokens.items()}
        # mask eos token
        eos_indices = [i for i, x in enumerate(tokens["input_ids"]) if x == self.tokenizer.eos_token_id]
        tokens["attention_mask"] = [0 if i in eos_indices else m for i, m in enumerate(tokens["attention_mask"])]
        tokens["labels"] = [-100 if i in eos_indices else l for i, l in enumerate(tokens["labels"])]

        # add eos_token
        tokens["input_ids"] = tokens["input_ids"] + [self.tokenizer.eos_token_id]
        tokens["attention_mask"] = tokens["attention_mask"] + [1]
        tokens["labels"] = tokens["labels"] + [self.tokenizer.eos_token_id]
        tokens = {k: v[: self.max_seq_length] for k, v in tokens.items()}
        tokens["img_path"] = image
        return tokens

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        signature_columns = ["input_ids", "labels", "attention_mask", "img_path"]
        return dataset.map(
            self.tokenize_row,
            batched=False,
            num_proc=self.dataset_num_proc,
            remove_columns=set(dataset.column_names) - set(signature_columns),
        )

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM
import os
from functools import wraps
from abc import ABC, abstractmethod
from loguru import logger
from trl import AutoModelForCausalLMWithValueHead
from peft import prepare_model_for_kbit_training, get_peft_model


class VLRewardModel(nn.Module, ABC):
    def __init__(self, base_model: PreTrainedModel, rm_head: nn.Linear | None = None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        hidden_size = self.base_model.config.hidden_size
        if rm_head is None:
            rm_head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(rm_head.bias)
        device = next(base_model.parameters()).device
        self.rm_head = rm_head.to(device)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(
            logits
        )  # Hacking to make sure every parameter is used in the backward pass. Copy from Llava-RLHF
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.rm_head(last_hidden_state_at_the_end)
        return (rewards,)  # return a tuple to be compatiable with RewardTrainer's compute_loss

    def save_pretrained(
        self, save_directory, is_main_process: bool = True, state_dict: dict | None = None, *args, **kwargs
    ):
        if state_dict:
            state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items() if k.startswith("base_model.")}
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        torch.save(self.rm_head.state_dict(), os.path.join(save_directory, "rm_head.bin"))

    @classmethod
    @abstractmethod
    @wraps(AutoModelForCausalLM.from_pretrained)
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        raise NotImplementedError

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # this method is used by PEFT
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def _get_reward_head_from_pretrained(self, pretrained_model_name_or_path, hidden_size):
        rm_head_path = os.path.join(pretrained_model_name_or_path, "rm_head.bin")
        if os.path.exists(rm_head_path):
            rm_head = nn.Linear(hidden_size, 1)
            rm_head.load_state_dict(torch.load(rm_head_path))
        else:
            logger.info(
                f"""No rm_head.bin found at {pretrained_model_name_or_path}.
                Make sure you are initializing reward model from a base model."""
            )
            rm_head = None
        return rm_head


class VLModelWithValueHead(AutoModelForCausalLMWithValueHead, ABC):
    supported_rm_modules = ("rm_head",)

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.value_adapter_name = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        is_loaded_in_8bit = getattr(pretrained_model_name_or_path, "is_loaded_in_8bit", False)
        is_loaded_in_4bit = getattr(pretrained_model_name_or_path, "is_loaded_in_4bit", False)
        value_adapter_config = kwargs.pop("value_adapter_config", None)
        value_adapter_name = kwargs.pop("value_adapter_name", "value_adapter")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        pretrained_model = model.pretrained_model
        if value_adapter_config is not None:
            if model.is_peft_model:
                pretrained_model.add_adapter(peft_config=value_adapter_config, adapter_name=value_adapter_name)
            else:
                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(
                        pretrained_model,
                        # **peft_quantization_kwargs, #TODO: add this to the config
                    )
                pretrained_model = get_peft_model(
                    pretrained_model, value_adapter_config, adapter_name=value_adapter_name
                )
            model.value_adapter_name = value_adapter_name
        return model

    def enable_input_require_grads(self):
        return self.pretrained_model.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.pretrained_model.get_input_embeddings()

    def compute_reward_score(self, input_ids, attention_mask=None, **kwargs):
        r"""
        Computes the reward score for a given input. The method has first to enable the adapter
        and then compute the reward score. After that the model disables the reward modeling
        adapter and enables the default ppo adapter again.
        """
        if not self.supports_rm_adapter:
            raise ValueError("This model does not support reward modeling adapter.")

        # enable rm adapter
        self.pretrained_model.set_adapter(self.rm_adapter_name)
        self.pretrained_model.eval()

        with torch.no_grad():
            base_model_output = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

            last_hidden_states = base_model_output.hidden_states[-1]
            last_hidden_state_at_the_end = last_hidden_states[
                :, -1, :
            ]  # add this to be compatiable with our VLRewardModel
            scores = self.score(last_hidden_state_at_the_end)

        self.pretrained_model.set_adapter(self.policy_adapter_name)
        self.pretrained_model.eval()

        return scores

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")
        if self.value_adapter_name is not None:
            self.pretrained_model.set_adapter(self.value_adapter_name)
            self.pretrained_model.eval()
            value_model_output = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

            self.pretrained_model.set_adapter(self.policy_adapter_name)
            self.pretrained_model.eval()
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        if self.value_adapter_name is None:
            value_model_output = base_model_output
        last_hidden_state = value_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

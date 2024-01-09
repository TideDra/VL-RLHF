from transformers import LlavaForConditionalGeneration
import torch
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel,AutoModelForCausalLM
import os
from functools import wraps
from abc import ABC,abstractclassmethod
from loguru import logger
from trl import AutoModelForCausalLMWithValueHead
from peft import prepare_model_for_kbit_training, get_peft_model
@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class LlavaRLOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    logp_labels: Optional[torch.FloatTensor] = None


class LlavaForRL(LlavaForConditionalGeneration):
    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, position_ids, logp_labels
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        #* hack for getting final_logp_labels
        # 6. Create the final logp_labels by replacing the image tokens with label_pad_token_id
        if logp_labels is not None:
            final_logp_labels = torch.full( (batch_size, max_embed_dim), fill_value=self.config.label_pad_token_id, dtype=logp_labels.dtype, device=target_device)  
            # Copy over the non-image token IDs to their new positions  
            final_logp_labels[batch_indices, text_to_overwrite] = logp_labels[batch_indices, non_image_indices]
        else:
            final_logp_labels = None
        return final_embedding, final_attention_mask, position_ids,final_logp_labels
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logp_labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, LlavaRLOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, position_ids,final_logp_labels = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, position_ids, logp_labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[batch_index, non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaRLOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logp_labels=final_logp_labels
        )

class VLRewardModel(nn.Module,ABC):
    def __init__(self, base_model:PreTrainedModel, rm_head:nn.Linear|None = None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        hidden_size = self.base_model.config.hidden_size
        if rm_head is None:
            rm_head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(rm_head.bias)
        device = next(base_model.parameters()).device
        self.rm_head = rm_head.to(device)
    
    def forward(self,input_ids,attention_mask,*args,**kwargs):
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask,*args,**kwargs)
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits) #Hacking to make sure every parameter is used in the backward pass. Copy from Llava-RLHF
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.rm_head(last_hidden_state_at_the_end)
        return (rewards,) # return a tuple to be compatiable with RewardTrainer's compute_loss
    
    def save_pretrained(self,save_directory,is_main_process:bool = True, state_dict:dict|None=None,*args,**kwargs):
        if state_dict:
            state_dict = {k.replace('base_model.',''):v for k,v in state_dict.items() if k.startswith('base_model.')}
        self.base_model.save_pretrained(save_directory,*args,**kwargs)
        torch.save(self.rm_head.state_dict(),os.path.join(save_directory,'rm_head.bin'))
    
    @abstractclassmethod
    @wraps(AutoModelForCausalLM.from_pretrained)
    def from_pretrained(cls,pretrained_model_name_or_path,*args,**kwargs):
        raise NotImplementedError

    def prepare_inputs_for_generation(self,*args,**kwargs):
        # this method is used by PEFT
        return self.base_model.prepare_inputs_for_generation(*args,**kwargs)

    def gradient_checkpointing_enable(self,*args,**kwargs):
        return self.base_model.gradient_checkpointing_enable(*args,**kwargs)

    def _get_reward_head_from_pretrained(self,pretrained_model_name_or_path,hidden_size):
        rm_head_path = os.path.join(pretrained_model_name_or_path,'rm_head.bin')
        if os.path.exists(rm_head_path):
            rm_head = nn.Linear(hidden_size, 1)
            rm_head.load_state_dict(torch.load(rm_head_path))
        else:
            logger.info(f'No rm_head.bin found at {pretrained_model_name_or_path}. Make sure you are initializing reward model from a base model.')
            rm_head = None
        return rm_head
    
class LlavaRewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*args,**kwargs):
        base_model = LlavaForRL.from_pretrained(pretrained_model_name_or_path,*args,**kwargs)
        rm_head = cls._get_reward_head_from_pretrained(cls,pretrained_model_name_or_path,base_model.config.hidden_size)
        return cls(base_model,rm_head)

class QwenVLRewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*args,**kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,*args,**kwargs)
        rm_head = cls._get_reward_head_from_pretrained(cls,pretrained_model_name_or_path,base_model.config.hidden_size)
        return cls(base_model,rm_head)

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
        value_adapter_name = kwargs.pop("value_adapter_name", 'value_adapter')
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        pretrained_model = model.pretrained_model
        if value_adapter_config is not None:
            if model.is_peft_model:
                pretrained_model.add_adapter(peft_config=value_adapter_config, adapter_name=value_adapter_name)
            else:
                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(
                        pretrained_model,
                        #**peft_quantization_kwargs, #TODO: add this to the config
                    )
                pretrained_model = get_peft_model(pretrained_model, value_adapter_config,adapter_name=value_adapter_name)
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
            last_hidden_state_at_the_end = last_hidden_states[:, -1, :] # add this to be compatiable with our VLRewardModel
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
        if self.value_adapter_name is not None:
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


class LlavaWithValueHead(VLModelWithValueHead):
    transformers_parent_class = LlavaForRL

class QwenVLWithValueHead(VLModelWithValueHead):
    transformers_parent_class = AutoModelForCausalLM
from typing import List, Dict, Optional, Union, Literal, Any, Tuple
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import InterpolationMode
from ...base.collator import (
    VLDPODataCollatorWithPadding,
    VLSFTDataCollatorWithPadding,
    VLRMDataCollatorWithPadding,
    VLPPODataCollator,
)
from ...base.trainer import VLDPOTrainer, VLPPOTrainer, VLSFTTrainer, VLRMTrainer
from ...base.processor import VLProcessor, VLChatTemplate
from ...base.model import VLRewardModel, VLModelWithValueHead
from dataclasses import dataclass
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..utils import ModelCoreMapper
from .modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from ...utils.common import flatten_list


@dataclass
class CausalLMOutputWithPastandImgPos(CausalLMOutputWithPast):
    image_position_map: torch.Tensor = None
    labels: torch.LongTensor = None


class InternLMXC2ForRL(InternLMXComposer2ForCausalLM):
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens. # noqa:E501
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
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
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
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"  # noqa:E501
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."  # noqa:E501
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids, image_to_overwrite

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        im_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        # we don't resize embeddings here, and <ImageHere> is not in the embedding matrix. We will replace it with padding token. # noqa:E501
        fake_input_ids = torch.where(
            input_ids == self.config.image_token_index, self.config.pad_token_id, input_ids
        ).to(input_ids.device)
        assert pixel_values is not None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(fake_input_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_features = self.encode_img(pixel_values)
                inputs_embeds, attention_mask, labels, position_ids, im_mask = (
                    self._merge_input_ids_with_image_features(
                        image_features, inputs_embeds, input_ids, attention_mask, labels
                    )
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

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/ 28032#issuecomment-1863691941 # noqa:E501
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        if im_mask is None and inputs_embeds is not None:
            im_mask = torch.zeros(inputs_embeds.shape[:2]).to(inputs_embeds.device)
            im_mask = im_mask.bool()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            input_ids = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastandImgPos(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
            image_position_map=im_mask,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, im_mask=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            im_mask=im_mask,
            **kwargs,
        )
        model_inputs["pixel_values"] = kwargs.get("pixel_values", None)
        return model_inputs

    @property
    def default_lora_target(self):
        return ["attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]

    def get_vision_tower(self):
        return self.vit

    def freeze_vision_tower(self):
        self.vit.requires_grad_(False)
        self.vision_proj.requires_grad_(False)
        self.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity()

    def prepare_default_generation_kwargs(self, generation_config):
        generation_config.do_sample = False
        generation_config.eos_token_id = 2

        # generation_config.top_p = 0.3
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, stop_words_ids: list, pad_token_id: int) -> None:
                super().__init__()
                self.stop_words_ids = stop_words_ids
                self.pad_token_id = pad_token_id

            def __call__(self, input_ids: torch.LongTensor, *args, **kwargs):
                is_done = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
                for stop_word_id in self.stop_words_ids:
                    is_done |= input_ids[:, -1] == stop_word_id
                input_ids[:, -1] = input_ids[:, -1] * (~is_done) + self.pad_token_id * is_done
                return is_done

        stopping_criteria_list = StoppingCriteriaList([KeywordsStoppingCriteria([92542], 2)])  # [UNUSED_TOKEN_145]
        kwargs = dict(generation_config=generation_config, stopping_criteria=stopping_criteria_list)
        return kwargs


class InternLMXC2RewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        base_model = InternLMXComposer2ForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        rm_head = cls._get_reward_head_from_pretrained(
            cls, pretrained_model_name_or_path, base_model.config.hidden_size
        )
        return cls(base_model, rm_head)


class InternLMXC2WithValueHead(VLModelWithValueHead):
    transformers_parent_class = InternLMXComposer2ForCausalLM


class InternLMXC2Processor(VLProcessor):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )
        self.__tokenizer.add_tokens("<ImageHere>", special_tokens=True)

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def chat_template(self):
        template = VLChatTemplate(
            system_begin="<s>[UNUSED_TOKEN_146]system\n",
            system_end="[UNUSED_TOKEN_145]\n",
            user_begin="[UNUSED_TOKEN_146]user\n",
            user_end="[UNUSED_TOKEN_145]\n",
            assistant_begin="[UNUSED_TOKEN_146]assistant\n",
            assistant_end="[UNUSED_TOKEN_145]\n",
            image_placeholder="<ImageHere>",
        )
        return template

    @property
    def __single_image_processor(self):
        processor = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.img_size, self.config.img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        return processor

    def image_processor(self, images):
        return torch.stack([self.__single_image_processor(img) for img in images])

    def save_pretrained(self, output_dir):
        return None

    def process_batch_conv(self, sources, system_message=None, add_end_for_empty_value=False):
        if not isinstance(sources, list) or not isinstance(sources[0], list):
            raise ValueError("sources must be a batch of conversations, eg. List[List[Dict]]")
        role_begin = {"user": self.chat_template.user_begin, "assistant": self.chat_template.assistant_begin}
        role_end = {"user": self.chat_template.user_end, "assistant": self.chat_template.assistant_end}
        raw_texts = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        for source in sources:
            __raw_text = """<s>[UNUSED_TOKEN_146]system\nYou are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n-InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n-InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n-InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.[UNUSED_TOKEN_145]\n"""  # noqa:E501
            attention_masks = []
            labels = []
            previous_len = 0
            for idx, sentence in enumerate(source):
                begin = role_begin[sentence["from"]]
                end = role_end[sentence["from"]]
                extend_text = (
                    begin + sentence["value"] + (end if sentence["value"] != "" or add_end_for_empty_value else "")
                )
                __raw_text += extend_text
                text_tokens = self.tokenizer(sentence["value"], padding=False, add_special_tokens=(idx == 0))
                full_tokens = self.tokenizer(extend_text, padding=False, add_special_tokens=(idx == 0))
                current_tokens = self.tokenizer(__raw_text)
                input_ids = current_tokens["input_ids"]
                attention_masks = current_tokens["attention_mask"]
                extend_len = len(input_ids) - previous_len
                previous_len = len(input_ids)
                labels.extend([-100] * extend_len)
                if sentence["from"] == "assistant" and len(text_tokens["input_ids"]) != 0:
                    if extend_len < len(full_tokens["input_ids"]):
                        # merging happens
                        labels[-extend_len:] = text_tokens["input_ids"][-extend_len:]
                    else:
                        labels[-len(text_tokens["input_ids"]) :] = text_tokens["input_ids"]
            labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_masks)]
            assert len(input_ids) == len(attention_masks) == len(labels)
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_masks)
            batch_labels.append(labels)
            raw_texts.append(__raw_text)
        return {
            "prompt": None,
            "answer": None,
            "full": dict(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels),
            "raw_str": raw_texts,
        }

    @staticmethod
    def format_multimodal_prompt(prompt: str, img_paths: Optional[Union[List[str], str]] = None):
        if img_paths is None:
            return prompt
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        if len(img_paths) == 1 and "<image>" not in prompt:
            return "<ImageHere>" + prompt
        else:
            assert prompt.count("<image>") == len(
                img_paths
            ), f"The number of given image ({len(img_paths)}) does not match the number of image placeholders in the prompt: {prompt}"  # noqa:E501
            prompt = prompt.replace("<image>", "<ImageHere>")
            return prompt

    @staticmethod
    def remove_image_placeholder(prompt: str):
        return prompt.replace("<ImageHere>", "")

    @staticmethod
    def is_multimodal_prompt_valid(prompt: str):
        return "<ImageHere>" in prompt

    def train(self):
        self.tokenizer.padding_side = "right"

    def infer(self):
        self.tokenizer.padding_side = "left"  # However, we pad tokens by ourselves.

    def __call__(
        self,
        texts: str | List[str] = None,
        convs: List[dict] = None,
        images_path: Optional[List[str | List[str]]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "left",
        check_format: bool = True,
    ):
        inputs = super().__call__(texts, convs, images_path, padding, padding_side, check_format)
        if images_path is not None:
            images = [Image.open(img_path).convert("RGB") for img_path in flatten_list(images_path)]
            inputs["pixel_values"] = self.image_processor(images)
        return inputs


class InternLMXC2DPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["img_input_dict"] = dict(pixel_values=self.processor.image_processor(images=imgs))
        return padded_batch


@dataclass
class InternLMXC2SFTDataCollatorWithPadding(VLSFTDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["pixel_values"] = self.processor.image_processor(images=imgs)
        padded_batch.pop("img_path")
        return padded_batch


@dataclass
class InternLMXC2RMDataCollatorWithPadding(VLRMDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["pixel_values"] = self.processor.image_processor(images=imgs)
        return padded_batch


@dataclass
class InternLMXC2PPODataCollator(VLPPODataCollator):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in batch["img_path"]]
        batch["pixel_values"] = self.processor.image_processor(images=imgs)
        return batch


class InternLMXC2DPOTrainer(VLDPOTrainer): ...


class InternLMXC2PPOTrainer(VLPPOTrainer): ...


class InternLMXC2RMTrainer(VLRMTrainer): ...


class InternLMXC2SFTTRainer(VLSFTTrainer): ...


core_mapper = ModelCoreMapper(
    model=InternLMXC2ForRL,
    processor=InternLMXC2Processor,
    dpo_collator=InternLMXC2DPODataCollatorWithPadding,
    dpo_trainer=InternLMXC2DPOTrainer,
    reward_model=InternLMXC2RewardModel,
    value_model=InternLMXC2WithValueHead,
    reward_collator=InternLMXC2RMDataCollatorWithPadding,
    reward_trainer=InternLMXC2RMTrainer,
    sft_collator=InternLMXC2SFTDataCollatorWithPadding,
    sft_trainer=InternLMXC2SFTTRainer,
    ppo_collator=InternLMXC2PPODataCollator,
    ppo_trainer=InternLMXC2PPOTrainer,
)

from typing import List, Dict, Any, Optional, Union, Tuple, Literal
import torch
import torch.nn as nn
from PIL import Image
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
from transformers.modeling_outputs import ModelOutput
from transformers import LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
from ..utils import ModelCoreMapper
from ...utils.common import flatten_list
from transformers import AutoConfig


@dataclass
class LlavaNextForRLOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[torch.FloatTensor] = None
    image_position_map: Optional[torch.LongTensor] = None


class LlavaNextForRL(LlavaNextForConditionalGeneration):

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True
            if batch_size > 1:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."  # noqa:E501
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=feature_lens.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens. # noqa:E501
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:] # noqa:E501
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_labels = None
        if labels is not None:
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
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

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835) # noqa:E501
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, image_to_overwrite

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, LlavaNextForRLOutputWithPast]:

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
        image_position_map = None
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                # ! infer image_num_patches from image_sizes
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.config.image_grid_pinpoints,
                        patch_size=self.config.vision_config.image_size,
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                image_features = self.multi_modal_projector(selected_image_feature)

                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, image_position_map = (
                    self._merge_input_ids_with_image_features(
                        image_features,
                        feature_lens,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        position_ids,
                        labels=labels,
                    )
                )

            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941 # noqa:E501
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
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

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextForRLOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
            image_position_map=image_position_map,
        )

    @property
    def default_lora_target(self):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["multi_modal_projector", "vision_tower", "vision_resampler"]
        for name, module in self.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    def get_vision_tower(self):
        return self.vision_tower

    def freeze_vision_tower(self):
        self.vision_tower.requires_grad_(False)

    def prepare_default_generation_kwargs(self, generation_config):
        generation_config.max_new_tokens = 1024
        generation_config.do_sample = False
        kwargs = dict(generation_config=generation_config)
        return kwargs


class LlavaNextRewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        base_model = LlavaNextForRL.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        rm_head = cls._get_reward_head_from_pretrained(
            cls, pretrained_model_name_or_path, base_model.config.hidden_size
        )
        return cls(base_model, rm_head)


class LlavaNextWithValueHead(VLModelWithValueHead):
    transformers_parent_class = LlavaNextForRL


class LlavaNextProcessor(VLProcessor):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.processor = transformers.LlavaNextProcessor.from_pretrained(model_name_or_path, **kwargs)
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def chat_template(self):
        if "mistral" in self.config.text_config._name_or_path:
            template = VLChatTemplate(
                system_begin=None,
                system_end=None,
                user_begin="[INST] ",
                user_end=" [/INST]",
                assistant_begin="",
                assistant_end="",
                image_placeholder="<image>\n",
            )
        elif "vicuna" in self.config.text_config._name_or_path:
            template = VLChatTemplate(
                system_begin=None,
                system_end=None,
                user_begin="USER: ",
                user_end="",
                assistant_begin="ASSISTANT: ",
                assistant_end="",
                image_placeholder="<image>\n",
            )
        return template

    @property
    def image_processor(self):
        return self.processor.image_processor

    def save_pretrained(self, output_dir):
        return self.processor.save_pretrained(output_dir)

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
            __raw_text = (
                "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "  # noqa:E501
                if "vicuna" in self.config.text_config._name_or_path
                else ""
            )
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
                current_tokens = self.tokenizer(__raw_text)
                input_ids = current_tokens["input_ids"]
                attention_masks = current_tokens["attention_mask"]
                extend_len = len(input_ids) - previous_len
                previous_len = len(input_ids)
                labels.extend([-100] * extend_len)
                if sentence["from"] == "assistant" and len(text_tokens["input_ids"]) != 0:
                    target_len = min([extend_len, len(text_tokens["input_ids"]), len(labels)])
                    labels[-target_len:] = text_tokens["input_ids"][-target_len:]

            labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_masks)]
            assert (
                len(input_ids) == len(attention_masks) == len(labels)
            ), f"input_ids:{len(input_ids)}, attention_masks:{len(attention_masks)}, labels:{len(labels)}"
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
            return "<image>\n" + prompt
        else:
            assert prompt.count("<image>") == len(
                img_paths
            ), f"The number of given image ({len(img_paths)}) does not match the number of image placeholders in the prompt: {prompt}"  # noqa:E501
            prompt = prompt.replace("<image>", "<image>\n")
            return prompt

    @staticmethod
    def remove_image_placeholder(prompt: str):
        return prompt.replace("<image>\n", "")

    @staticmethod
    def is_multimodal_prompt_valid(prompt: str):
        return "<image>\n" in prompt

    def train(self):
        pass

    def infer(self):
        self.tokenizer.pad_token = self.tokenizer.bos_token

    def __call__(
        self,
        texts: str | List[str] = None,
        convs: List[Dict] = None,
        images_path: Optional[List[str | List[str]]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "left",
        check_format: bool = True,
    ):
        inputs = super().__call__(texts, convs, images_path, padding, padding_side, check_format)
        if images_path is not None:
            images = [Image.open(img_path).convert("RGB") for img_path in flatten_list(images_path)]
            inputs.update(self.image_processor(images=images, return_tensors="pt"))
        return inputs


class LlavaNextDPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["img_input_dict"] = self.processor.image_processor(images=imgs, return_tensors="pt")
        return padded_batch


@dataclass
class LlavaNextSFTDataCollatorWithPadding(VLSFTDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch.update(self.image_processor(images=imgs, return_tensors="pt"))
        padded_batch.pop("img_path")
        return padded_batch


@dataclass
class LlavaNextRMDataCollatorWithPadding(VLRMDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch.update(self.image_processor(images=imgs, return_tensors="pt"))
        padded_batch.pop("img_path")
        return padded_batch


@dataclass
class LlavaNextPPODataCollator(VLPPODataCollator):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in batch["img_path"]]
        batch.update(self.image_processor(images=imgs, return_tensors="pt"))
        batch.pop("img_path")
        return batch


class LlavaNextDPOTrainer(VLDPOTrainer): ...


class LlavaNextPPOTrainer(VLPPOTrainer): ...


class LlavaNextRMTrainer(VLRMTrainer): ...


class LlavaNextSFTTRainer(VLSFTTrainer): ...


core_mapper = ModelCoreMapper(
    model=LlavaNextForRL,
    processor=LlavaNextProcessor,
    dpo_collator=LlavaNextDPODataCollatorWithPadding,
    dpo_trainer=LlavaNextDPOTrainer,
    reward_model=LlavaNextRewardModel,
    value_model=LlavaNextWithValueHead,
    reward_collator=LlavaNextRMDataCollatorWithPadding,
    reward_trainer=LlavaNextRMTrainer,
    sft_collator=LlavaNextSFTDataCollatorWithPadding,
    sft_trainer=LlavaNextSFTTRainer,
    ppo_collator=LlavaNextPPODataCollator,
    ppo_trainer=LlavaNextPPOTrainer,
)

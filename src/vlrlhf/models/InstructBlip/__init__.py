from typing import List, Dict, Any, Optional, Union, Tuple, Literal
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput

from vlrlhf.base.model import VLModelWithValueHead, VLRewardModel
from ...base.collator import (
    VLDPODataCollatorWithPadding,
    VLSFTDataCollatorWithPadding,
    VLRMDataCollatorWithPadding,
    VLPPODataCollator,
)
from ...base.trainer import VLDPOTrainer, VLPPOTrainer, VLSFTTrainer, VLRMTrainer
from ...base.processor import VLProcessor, VLChatTemplate
from dataclasses import dataclass
import transformers
from transformers import InstructBlipForConditionalGeneration
from transformers import (
    PreTrainedModel,
)
from ..utils import ModelCoreMapper


@dataclass
class InstructBlipForRLModelOutput(InstructBlipForConditionalGenerationModelOutput):
    image_position_map: torch.Tensor = None
    attentions: torch.Tensor = None


class InstructBlipForRL(InstructBlipForConditionalGeneration):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: torch.LongTensor | None = None,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
    ) -> Tuple | InstructBlipForConditionalGenerationModelOutput:
        output = super().forward(
            pixel_values,
            qformer_input_ids,
            qformer_attention_mask,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            output_attentions,
            output_hidden_states,
            labels,
            return_dict,
        )
        img_token_num = self.query_tokens.size(1)
        img_pos = torch.full(output.language_model_outputs.logits.shape[:2], False, dtype=torch.bool)
        img_pos[:, :img_token_num] = True
        attentions = output.language_model_outputs.attentions if output_attentions else None
        return InstructBlipForRLModelOutput(
            loss=output.loss,
            logits=output.logits,
            vision_outputs=output.vision_outputs,
            qformer_outputs=output.qformer_outputs,
            language_model_outputs=output.language_model_outputs,
            image_position_map=img_pos,
            attentions=attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # the InstructBLIP authors used inconsistent tokenizer/model files during training,
        # with the tokenizer's bos token being set to </s> which has ID=2,
        # whereas the model's text config has bos token id = 0
        if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
            if isinstance(outputs, torch.Tensor):
                outputs[outputs == 0] = 2
            else:
                outputs.sequences[outputs.sequences == 0] = 2

        return outputs

    @property
    def default_lora_target(self):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["vision_model", "language_projection", "qformer"]
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
        return self.vision_model

    def freeze_vision_tower(self):
        self.vision_model.requires_grad_(False)

    def prepare_default_generation_kwargs(self, generation_config):
        generation_config.temperature = 0.2
        generation_config.do_sample = True
        generation_config.max_length = 1024
        kwargs = dict(generation_config=generation_config)
        return kwargs


class InstructBlipRewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        base_model = InstructBlipForRL.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        rm_head = cls._get_reward_head_from_pretrained(
            cls, pretrained_model_name_or_path, base_model.config.hidden_size
        )
        return cls(base_model, rm_head)


class InstructBlipWithValueHead(VLModelWithValueHead):
    transformers_parent_class = InstructBlipForRL


class InstructBlipProcessor(VLProcessor):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.processor = transformers.InstructBlipProcessor.from_pretrained(model_name_or_path, **kwargs)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def qformer_tokenizer(self):
        return self.processor.qformer_tokenizer

    @property
    def image_processor(self):
        return self.processor.image_processor

    @property
    def chat_template(self):
        template = VLChatTemplate(
            system_begin=None,
            system_end=None,
            user_begin="",
            user_end="",
            assistant_begin="",
            assistant_end="",
            image_placeholder="",
        )
        return template

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
            __raw_text = ""
            attention_masks = []
            labels = []
            previous_len = 0
            if len(source) > 2:
                raise ValueError("InstructBlip does not support multi-turn conversation.")
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
                    target_len = max(min([extend_len, len(text_tokens["input_ids"]), len(labels)]), 1)
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
        # currently not support multi images
        return prompt

    @staticmethod
    def remove_image_placeholder(prompt: str):
        return prompt

    @staticmethod
    def is_multimodal_prompt_valid(prompt: str):
        return True

    def train(self):
        pass

    def infer(self):
        self.tokenizer.pad_token = self.tokenizer.bos_token

    def __call__(
        self,
        texts: str | List[str] = None,
        convs: List[dict] = None,
        images_path: Optional[List[str]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "left",
        check_format: bool = True,
    ):
        inputs = super().__call__(texts, convs, images_path, padding, padding_side, check_format)
        if convs and texts is None:
            texts = [conv[0]["value"] for conv in convs]
        qformer_inputs = self.qformer_tokenizer(text=texts, padding=padding, return_tensors="pt")
        inputs["qformer_input_ids"] = qformer_inputs["input_ids"][:, :512]
        inputs["qformer_attention_mask"] = qformer_inputs["attention_mask"][:, :512]
        images = [Image.open(img_path).convert("RGB") for img_path in images_path]
        inputs["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return inputs


class InstructBlipDPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        img_input_dict = {}
        img_input_dict["pixel_values"] = self.processor.image_processor(images=imgs, return_tensors="pt")[
            "pixel_values"
        ]
        qformer_tokenizer = self.processor.qformer_tokenizer
        padded_qformer_input_ids = padded_batch.pop("qformer_input_ids")
        qformer_pad_id = torch.ones_like(padded_qformer_input_ids) * qformer_tokenizer.pad_token_id
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids > len(qformer_tokenizer) - 1, qformer_pad_id, padded_qformer_input_ids
        )
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids == self.pad_token_id, qformer_pad_id, padded_qformer_input_ids
        )
        img_input_dict["qformer_input_ids"] = padded_qformer_input_ids
        img_input_dict["qformer_attention_mask"] = padded_batch.pop("qformer_attention_mask")
        padded_batch["img_input_dict"] = img_input_dict
        return padded_batch


@dataclass
class InstructBlipSFTDataCollatorWithPadding(VLSFTDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        qformer_tokenizer = self.processor.qformer_tokenizer
        for k in features[0].keys():
            if k == "img_path":
                padded_batch[k] = [ex[k] for ex in features]
                continue
            to_pad = [torch.LongTensor(ex[k]) for ex in features]
            if k == "input_ids":
                padding_value = self.pad_token_id
            elif k == "qformer_input_ids":
                padding_value = qformer_tokenizer.pad_token_id
            elif k == "labels":
                padding_value = self.label_pad_token_id
            elif k.endswith("attention_mask"):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["pixel_values"] = self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        padded_batch.pop("img_path")
        padded_qformer_input_ids = padded_batch["qformer_input_ids"]
        qformer_pad_id = torch.ones_like(padded_qformer_input_ids) * qformer_tokenizer.pad_token_id
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids > len(qformer_tokenizer) - 1, qformer_pad_id, padded_qformer_input_ids
        )
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids == self.pad_token_id, qformer_pad_id, padded_qformer_input_ids
        )
        padded_batch["qformer_input_ids"] = padded_qformer_input_ids
        return padded_batch


@dataclass
class InstructBlipRMDataCollatorWithPadding(VLRMDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["pixel_values"] = self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        qformer_tokenizer = self.processor.qformer_tokenizer
        padded_qformer_input_ids = padded_batch["qformer_input_ids"]
        qformer_pad_id = torch.ones_like(padded_qformer_input_ids) * qformer_tokenizer.pad_token_id
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids > len(qformer_tokenizer) - 1, qformer_pad_id, padded_qformer_input_ids
        )
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids == self.pad_token_id, qformer_pad_id, padded_qformer_input_ids
        )
        padded_batch["qformer_input_ids"] = padded_qformer_input_ids
        return padded_batch


@dataclass
class InstructBlipPPODataCollator(VLPPODataCollator):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["pixel_values"] = self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        qformer_tokenizer = self.processor.qformer_tokenizer
        padded_qformer_input_ids = padded_batch["qformer_input_ids"]
        qformer_pad_id = torch.ones_like(padded_qformer_input_ids) * qformer_tokenizer.pad_token_id
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids > len(qformer_tokenizer) - 1, qformer_pad_id, padded_qformer_input_ids
        )
        padded_qformer_input_ids = torch.where(
            padded_qformer_input_ids == self.pad_token_id, qformer_pad_id, padded_qformer_input_ids
        )
        padded_batch["qformer_input_ids"] = padded_qformer_input_ids
        return padded_batch


class InstructBlipDPOTrainer(VLDPOTrainer):
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        batch = super().tokenize_row(feature, model)
        prompt = feature["prompt"]
        qformer_tokens = self.processor.qformer_tokenizer(prompt)
        batch["qformer_input_ids"] = qformer_tokens["input_ids"]
        batch["qformer_attention_mask"] = qformer_tokens["attention_mask"]
        return batch


class InstructBlipPPOTrainer(VLPPOTrainer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PPO is not supported for InstructBlip currently.")


class InstructBlipRMTrainer(VLRMTrainer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Reward model is not supported for InstructBlip currently.")


class InstructBlipSFTTRainer(VLSFTTrainer):
    def tokenize_row(self, element):
        tokens = super().tokenize_row(element)
        conv = element["conversations"]
        prompt = conv[0]["value"]
        qformer_tokens = self.processor.qformer_tokenizer(prompt)
        tokens["qformer_input_ids"] = qformer_tokens["input_ids"]
        tokens["qformer_attention_mask"] = qformer_tokens["attention_mask"]
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
        signature_columns = [
            "input_ids",
            "labels",
            "attention_mask",
            "img_path",
            "qformer_input_ids",
            "qformer_attention_mask",
        ]
        return dataset.map(
            self.tokenize_row,
            batched=False,
            num_proc=self.dataset_num_proc,
            remove_columns=set(dataset.column_names) - set(signature_columns),
        )


core_mapper = ModelCoreMapper(
    model=InstructBlipForRL,
    processor=InstructBlipProcessor,
    dpo_collator=InstructBlipDPODataCollatorWithPadding,
    dpo_trainer=InstructBlipDPOTrainer,
    reward_model=InstructBlipRewardModel,
    value_model=InstructBlipWithValueHead,
    reward_collator=InstructBlipRMDataCollatorWithPadding,
    reward_trainer=InstructBlipRMTrainer,
    sft_collator=InstructBlipSFTDataCollatorWithPadding,
    sft_trainer=InstructBlipSFTTRainer,
    ppo_collator=InstructBlipPPODataCollator,
    ppo_trainer=InstructBlipPPOTrainer,
)

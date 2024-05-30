from typing import List, Dict, Optional, Union, Literal, Any
from torch.nn.modules import Module
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
import re
import transformers
from transformers import AutoModelForCausalLM
from transformers import (
    PreTrainedModel,
)
from transformers.trainer_pt_utils import LabelSmoother
from ..utils import ModelCoreMapper
from .modeling_qwen import QWenLMHeadModel


class QwenVLForRL(QWenLMHeadModel):

    @property
    def default_lora_target(self):
        return ["c_attn", "attn.c_proj", "w1", "w2"]

    def get_vision_tower(self):
        return self.transformer.visual

    def freeze_vision_tower(self):
        vision_tower = self.get_vision_tower()
        vision_tower.requires_grad_(False)
        if hasattr(vision_tower, "attn_pool"):  # follow Qwen-VL default setting
            vision_tower.attn_pool.requires_grad_(True)

    def prepare_default_generation_kwargs(self, generation_config):
        im_end_id = 151645
        im_start_id = 151644
        stop_words_ids = [[im_end_id], [im_start_id]]
        generation_config.stop_words_ids = stop_words_ids
        generation_config.do_sample = False
        kwargs = dict(generation_config=generation_config)
        return kwargs


class QwenVLRewardModel(VLRewardModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        rm_head = cls._get_reward_head_from_pretrained(
            cls, pretrained_model_name_or_path, base_model.config.hidden_size
        )
        return cls(base_model, rm_head)


class QwenVLWithValueHead(VLModelWithValueHead):
    transformers_parent_class = AutoModelForCausalLM


class QwenVLProcessor(VLProcessor):
    __multimodal_prompt_pattern = re.compile(r"<img>.+</img>\n")

    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def chat_template(self):
        return VLChatTemplate(
            system_begin="<|im_start|>system",
            system_end="<|im_end|>",
            user_begin="<|im_start|>user",
            user_end="<|im_end|>",
            assistant_begin="<|im_start|>assistant",
            assistant_end="<|im_end|>",
            image_placeholder="<img>",
        )

    @property
    def image_processor(self):
        return None

    def save_pretrained(self, output_dir):
        return None

    def process_batch_conv(
        self, sources: List[List[Dict]], system_message="You are a helpful assistant.", add_end_for_empty_value=False
    ):
        if not isinstance(sources, list) or not isinstance(sources[0], list):
            raise ValueError("sources must be a batch of conversations, eg. List[List[Dict]]")
        # FIXME: not know whether this code support multi-turn conversation
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        im_start = self.tokenizer.im_start_id
        im_end = self.tokenizer.im_end_id
        nl_tokens = self.tokenizer("\n").input_ids
        _system = self.tokenizer("system").input_ids + nl_tokens
        raw_texts = []
        # Apply prompt templates
        prompt_ids, prompt_targets = [], []
        answer_ids, answer_targets = [], []
        full_ids, full_targets = [], []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != roles["user"]:
                source = source[1:]

            input_id, target = [], []
            answer_id, answer_target = [], []
            prompt_id, prompt_target = [], []
            system = [im_start] + _system + self.tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
            _raw_text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                # * In generation mode, we only add these tokens
                _input_id = self.tokenizer(role).input_ids + nl_tokens
                _raw_text += f"{role}\n"
                if sentence["value"] != "" or add_end_for_empty_value:
                    _input_id += self.tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                    _raw_text += f"{sentence['value']}<|im_end|>\n"
                input_id += _input_id

                if role == "<|im_start|>user":
                    if sentence["value"] != "" or add_end_for_empty_value:
                        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
                    else:
                        # In generation mode, actually target is not used.
                        _target = [im_start] + [IGNORE_TOKEN_ID]
                    prompt_id.extend(input_id[:])
                    prompt_target.extend((target + _target)[:])
                elif role == "<|im_start|>assistant":
                    if sentence["value"] != "" or add_end_for_empty_value:
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids)
                            + _input_id[len(self.tokenizer(role).input_ids) + 1 : -2]
                            + [im_end]
                            + nl_tokens
                        )
                    else:
                        # In generation mode, actually target is not used.
                        _target = [im_start] + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids)
                    answer_id.extend(_input_id[:])
                    answer_target.extend(_target[:])
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            full_ids.append(input_id[:])
            full_targets.append(target[:])
            prompt_ids.append(prompt_id[:])
            prompt_targets.append(prompt_target[:])
            answer_ids.append(answer_id[:])
            answer_targets.append(answer_target[:])
            raw_texts.append(_raw_text)
            assert len(prompt_ids[-1]) == len(prompt_targets[-1])
            assert len(answer_ids[-1]) == len(answer_targets[-1])

        prompt_sequence_tokens = dict(
            input_ids=prompt_ids,
            labels=prompt_targets,
            attention_mask=[[int(id != self.tokenizer.pad_token_id) for id in ids] for ids in prompt_ids],
        )
        answer_sequence_tokens = dict(
            input_ids=answer_ids,
            labels=answer_targets,
            attention_mask=[[int(id != self.tokenizer.pad_token_id) for id in ids] for ids in answer_ids],
        )

        full_sequence_tokens = dict(
            input_ids=full_ids,
            labels=full_targets,
            attention_mask=[[int(id != self.tokenizer.pad_token_id) for id in ids] for ids in full_ids],
        )
        return {
            "prompt": prompt_sequence_tokens,
            "answer": answer_sequence_tokens,
            "full": full_sequence_tokens,
            "raw_str": raw_texts,
        }

    @staticmethod
    def format_multimodal_prompt(prompt: str, img_paths: Optional[Union[List[str], str]] = None):
        if img_paths is None:
            return prompt
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        if len(img_paths) == 1 and "<image>" not in prompt:
            prompt = f"Picture 1: <img>{img_paths[0]}</img>\n{prompt}"
        else:
            assert prompt.count("<image>") == len(
                img_paths
            ), f"The number of given image ({len(img_paths)}) does not match the number of image placeholders in the prompt: {prompt}"  # noqa:E501
            for img_path in img_paths:
                prompt = prompt.replace("<image>", f"<img>{img_path}</img>\n", 1)

        return prompt

    @staticmethod
    def remove_image_placeholder(prompt: str):
        return re.sub(QwenVLProcessor.__multimodal_prompt_pattern, "", prompt)

    @staticmethod
    def is_multimodal_prompt_valid(prompt: str):
        pattern = QwenVLProcessor.__multimodal_prompt_pattern
        return bool(pattern.search(prompt))

    def train(self):
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.tokenizer.eos_token_id = self.tokenizer.eod_id
        self.tokenizer.padding_side = "right"

    def infer(self):
        self.tokenizer.padding_side = "left"  # However, we pad tokens by ourselves.
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def __call__(
        self,
        texts: str | List[str] = None,
        convs: List[dict] = None,
        images_path: Optional[List[str | List[str]]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "left",
        check_format: bool = True,
    ):
        return super().__call__(texts, convs, images_path, padding, padding_side, check_format)


class QwenVLDPODataCollatorWithPadding(VLDPODataCollatorWithPadding): ...


@dataclass
class QwenVLSFTDataCollatorWithPadding(VLSFTDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)
        padded_batch.pop("img_path")
        return padded_batch


@dataclass
class QwenVLRMDataCollatorWithPadding(VLRMDataCollatorWithPadding): ...


@dataclass
class QwenVLPPODataCollator(VLPPODataCollator): ...


class QwenVLDPOTrainer(VLDPOTrainer):
    def tokenize_row(self, feature, model: PreTrainedModel | Module = None) -> Dict:
        # FIXME: try to use raw tokenize_raw code
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
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["labels"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["labels"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        for k, toks in {
            "chosen_": chosen_tokens,
            "rejected_": rejected_tokens,
            "prompt_": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        return batch


class QwenVLPPOTrainer(VLPPOTrainer): ...


class QwenVLRMTrainer(VLRMTrainer): ...


class QwenVLSFTTrainer(VLSFTTrainer): ...


core_mapper = ModelCoreMapper(
    model=QwenVLForRL,
    processor=QwenVLProcessor,
    dpo_collator=QwenVLDPODataCollatorWithPadding,
    dpo_trainer=QwenVLDPOTrainer,
    reward_model=QwenVLRewardModel,
    value_model=QwenVLWithValueHead,
    reward_collator=QwenVLRMDataCollatorWithPadding,
    reward_trainer=QwenVLRMTrainer,
    sft_collator=QwenVLSFTDataCollatorWithPadding,
    sft_trainer=QwenVLSFTTrainer,
    ppo_collator=QwenVLPPODataCollator,
    ppo_trainer=QwenVLPPOTrainer,
)

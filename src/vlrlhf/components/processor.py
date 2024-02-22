import transformers
from abc import abstractmethod, ABC,abstractclassmethod
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Dict, Union, Literal, Optional
import re
from loguru import logger
from PIL.Image import Image
from ..utils.common import pad_to_length
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizerBase

class VLProcessor(ABC):
    @abstractmethod
    def __init__(self, model_name_or_path, **kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def image_processor(self):
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(
        self,
        output_dir: str
    ):  # hack for some model uses unique processor, and we need to save it after training.
        raise NotImplementedError

    @abstractmethod
    def process_batch_conv(
        self, sources, system_message=None
    ) -> (
        dict
    ):  # tokenize a batch of conversations. We do not pad or return tensors here.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def format_multimodal_prompt(
        prompt: str, img_paths: Optional[Union[List[str],str]] = None
    ):  # * add image placeholder or source to prompt. must be used in VLDPOTrainer.tokenize_row
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def remove_image_placeholder(
        prompt: str
    ):  # remove image placeholder from prompt.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_multimodal_prompt_valid(
        prompt: str,
    ) -> bool:  # check if prompt contains image placeholder.
        raise NotImplementedError

    @staticmethod
    def make_single_turn_conv(prompt: str, answer: str = ""):
        return [
            {
                "from": "user",
                "value": prompt,
            },
            {
                "from": "assistant",
                "value": answer,
            },
        ]

    @abstractmethod
    def train(self):  # set tokenizer to train mode
        raise NotImplementedError

    @abstractmethod
    def infer(self):  # set tokenizer to inference mode
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        texts: Union[str, List[str]],
        images_path: Optional[List[str]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "right",
        check_format: bool = True
    ) -> BatchEncoding:
        """Porcess raw texts and images into model input tensors. Currently only support single turn conversation generation.

        Args:
            texts (Union[str,List[str]]): Raw texts with or without multimodal format. If you pass images but the texts are not in multimodal format, we will automatically add image placeholder to your texts. We recommend you to prepare multimodal prompts in advance. eg. "<image>\nWhat is the color of the cat?"
            images (Union[Image,List[Image]], optional): PIL images. Defaults to None.
            padding (bool, optional): Whether pad texts into the same length. Defaults to True.
            padding_side (Literal["right","left"], optional): Which side to pad. Defaults to "right".
        """
        #! this abstractmethod does not process images. Subclass should implement this method to process images.
        if images_path is not None and check_format:
            is_multimodal_prompt_valid = True
            for i in range(len(texts)):
                if not self.is_multimodal_prompt_valid(texts[i]):
                    is_multimodal_prompt_valid = False
                    texts[i] = self.format_multimodal_prompt(texts[i], images_path[i])
            if not is_multimodal_prompt_valid:
                logger.warning(
                    "You passed images, but your prompts are not in multimodal format. We will automatically add image placeholder to your prompts. We recommend you to prepare multimodal prompts in advance."
                )

        batch_conv = [self.make_single_turn_conv(text) for text in texts]
        tokenized_batch_conv = self.process_batch_conv(batch_conv)
        input_tokens = tokenized_batch_conv["full"]
        if padding:
            input_ids = input_tokens["input_ids"]
            pad_length = max([len(ids) for ids in input_ids])
            input_ids = [torch.tensor(ids) for ids in input_ids]
            input_ids = [
                pad_to_length(
                    ids,
                    pad_length,
                    self.tokenizer.pad_token_id,
                    padding_side=padding_side,
                )
                for ids in input_ids
            ]
            input_ids = torch.stack(input_ids)
            attention_mask = input_tokens["attention_mask"]
            attention_mask = [torch.tensor(mask) for mask in attention_mask]
            attention_mask = [
                pad_to_length(mask, pad_length, 0, padding_side=padding_side)
                for mask in attention_mask
            ]
            attention_mask = torch.stack(attention_mask)
        return BatchEncoding(
            data={"input_ids": input_ids, "attention_mask": attention_mask}
        )


class LlavaProcessor(VLProcessor):
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.processor = transformers.LlavaProcessor.from_pretrained(
            model_name_or_path, **kwargs
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def image_processor(self):
        return self.processor.image_processor

    def save_pretrained(self, output_dir):
        return self.processor.save_pretrained(output_dir)

    def process_batch_conv(self, sources, system_message):
        if not isinstance(sources,list) or not isinstance(sources[0],list):
            raise ValueError("sources must be a batch of conversations, eg. List[List[Dict]]")
        roles = {"user": "USER: ", "assistant": "ASSISTANT: "}
        raw_texts = []
        for source in sources:
            __raw_text = ""
            for sentence in source:
                role = roles[sentence["from"]]
                __raw_text += role + sentence["value"] + " "
            __raw_text = __raw_text.strip()
        raw_texts.append(__raw_text)
        return {
            # not process other tokens. They will be processed by tokenize_row. This method is just used to process conversation into correct format.
            "prompt": None,
            "answer": None,
            "full": self.tokenizer(raw_texts),
            "raw_str": raw_texts,
        }

    @staticmethod
    def format_multimodal_prompt(prompt: str, img_paths: Optional[Union[List[str],str]] = None):
        # currently not support multi images
        if img_paths is None:
            return prompt
        return "<image>\n" + prompt

    @staticmethod
    def remove_image_placeholder(prompt: str):
        return prompt.replace("<image>\n", "")

    @staticmethod
    def is_multimodal_prompt_valid(prompt: str):
        return "<image>\n" in prompt

    def train(self):
        self.tokenizer.pad_token = (
            self.tokenizer.unk_token
        )  #! not sure if this is correct

    def infer(self):
        self.tokenizer.pad_token = self.tokenizer.bos_token

    def __call__(
        self,
        texts: str | List[str],
        images_path: Optional[List[str]],
        padding: bool = True,
        padding_side: Literal["right", "left"] = "right",
        check_format: bool = True
    ):
        inputs = super().__call__(texts, images_path, padding, padding_side, check_format)
        images = [Image.open(img_path).convert("RGB") for img_path in images_path]
        inputs["pixel_values"] = self.image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"]
        return inputs


class QwenVLProcessor(VLProcessor):
    __multimodal_prompt_pattern = re.compile(r"Picture \d+: <img>.+</img>\n")

    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs
        )

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def image_processor(self):
        return None

    def save_pretrained(self,output_dir):
        return None

    def process_batch_conv(
        self, sources: List[List[Dict]], system_message="You are a helpful assistant."
    ):
        if not isinstance(sources,list) or not isinstance(sources[0],list):
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
            system = (
                [im_start]
                + _system
                + self.tokenizer(system_message).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += system
            target += (
                [im_start]
                + [IGNORE_TOKEN_ID] * (len(system) - 3)
                + [im_end]
                + nl_tokens
            )
            _raw_text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                # * In generation mode, we only add these tokens
                _input_id = self.tokenizer(role).input_ids + nl_tokens
                _raw_text += f"{role}\n"
                if sentence["value"] != "":
                    _input_id += (
                        self.tokenizer(sentence["value"]).input_ids
                        + [im_end]
                        + nl_tokens
                    )
                    _raw_text += f"{sentence['value']}<|im_end|>\n"
                input_id += _input_id

                if role == "<|im_start|>user":
                    if sentence["value"] != "":
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                            + [im_end]
                            + nl_tokens
                        )
                    else:
                        # In generation mode, actually target is not used.
                        _target = [im_start] + [IGNORE_TOKEN_ID]
                    prompt_ids.append(input_id[:])
                    prompt_targets.append((target + _target)[:])
                elif role == "<|im_start|>assistant":
                    if sentence["value"] != "":
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids)
                            + _input_id[len(self.tokenizer(role).input_ids) + 1 : -2]
                            + [im_end]
                            + nl_tokens
                        )
                    else:
                        # In generation mode, actually target is not used.
                        _target = [im_start] + [IGNORE_TOKEN_ID]* len(self.tokenizer(role).input_ids)
                    answer_ids.append(_input_id[:])
                    answer_targets.append(_target[:])
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            full_ids.append(input_id[:])
            full_targets.append(target[:])
            raw_texts.append(_raw_text)
            assert len(prompt_ids[-1]) == len(prompt_targets[-1])
            assert len(answer_ids[-1]) == len(answer_targets[-1])

        prompt_sequence_tokens = dict(
            input_ids=prompt_ids,
            labels=prompt_targets,
            attention_mask=[
                [int(id != self.tokenizer.pad_token_id) for id in ids] for ids in prompt_ids
            ],
        )
        answer_sequence_tokens = dict(
            input_ids=answer_ids,
            labels=answer_targets,
            attention_mask=[
                [int(id != self.tokenizer.pad_token_id) for id in ids] for ids in answer_ids
            ],
        )

        full_sequence_tokens = dict(
            input_ids=full_ids,
            labels=full_targets,
            attention_mask=[
                [int(id != self.tokenizer.pad_token_id) for id in ids] for ids in full_ids
            ],
        )
        return {
            "prompt": prompt_sequence_tokens,
            "answer": answer_sequence_tokens,
            "full": full_sequence_tokens,
            "raw_str": raw_texts,
        }

    @staticmethod
    def format_multimodal_prompt(prompt: str, img_paths: Optional[Union[List[str],str]] = None):
        if img_paths is None:
            return prompt
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        out = []
        for i, img_path in enumerate(img_paths):
            out.append(f"Picture {i + 1}: <img>{img_path}</img>\n")
        out.append(prompt.strip())
        return "".join(out)

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
        self.tokenizer.padding_side = 'left' # However, we pad tokens by ourselves.
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def __call__(
        self,
        texts: str | List[str],
        images_path: Optional[List[str]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "right",
        check_format: bool = True
    ):
        return super().__call__(texts, images_path, padding, padding_side,check_format)
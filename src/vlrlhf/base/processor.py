from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Union, Literal, Optional
from loguru import logger
from ..utils.common import pad_to_length
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizerBase


@dataclass
class VLChatTemplate:
    system_begin: str
    system_end: str
    user_begin: str
    user_end: str
    assistant_begin: str
    assistant_end: str
    image_placeholder: str


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
    def chat_template(self) -> VLChatTemplate:
        raise NotImplementedError

    @property
    @abstractmethod
    def image_processor(self):
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(
        self, output_dir: str
    ):  # hack for some model uses unique processor, and we need to save it after training.
        raise NotImplementedError

    @abstractmethod
    def process_batch_conv(
        self, sources, system_message=None, add_end_for_empty_value=False
    ) -> dict:  # tokenize a batch of conversations. We do not pad or return tensors here.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def format_multimodal_prompt(
        prompt: str, img_paths: Optional[Union[List[str], str]] = None
    ):  # * add image placeholder or source to prompt. must be used in VLDPOTrainer.tokenize_row
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def remove_image_placeholder(prompt: str):  # remove image placeholder from prompt.
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
        texts: str | List[str] = None,
        convs: List[dict] = None,
        images_path: Optional[List[str | List[str]]] = None,
        padding: bool = True,
        padding_side: Literal["right", "left"] = "left",
        check_format: bool = True,
    ) -> BatchEncoding:
        """Porcess raw texts and images into model input tensors.

        Args:
            texts (Union[str,List[str]]): Raw texts with or without multimodal format. If you pass images but the texts
                are not in multimodal format, we will automatically add image placeholder to your texts.
                We recommend you to prepare multimodal prompts in advance. eg. "<image>\nWhat is the color of the cat?"
            convs (List[dict]): List of formatted conversations. Each conversation is a list of turns.
                Each turn is a dict with keys "from" and "value".
                eg. [{"from":"user","value":"<image>\nWhat is the color of the cat?"},{"from":"assistant",
                "value":"The cat is white."}]
            images (Union[Image,List[Image]], optional): PIL images. Defaults to None.
            padding (bool, optional): Whether pad texts into the same length. Defaults to True.
            padding_side (Literal["right","left"], optional): Which side to pad. Defaults to "right".
        """
        assert texts is None or convs is None, "You can only pass texts or convs, not both."
        # ! this abstractmethod does not process images. Subclass should implement this method to process images.
        if texts:
            if images_path is not None and check_format:
                is_multimodal_prompt_valid = True
                for i in range(len(texts)):
                    if not self.is_multimodal_prompt_valid(texts[i]):
                        is_multimodal_prompt_valid = False
                        texts[i] = self.format_multimodal_prompt(texts[i], images_path[i])
                if not is_multimodal_prompt_valid:
                    logger.warning(
                        """You passed images, but your prompts are not in multimodal format. We will automatically add
                        image placeholder to your prompts. We recommend you to prepare multimodal prompts in advance."""
                    )

        batch_conv = [self.make_single_turn_conv(text) for text in texts] if texts else convs
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
            attention_mask = [pad_to_length(mask, pad_length, 0, padding_side=padding_side) for mask in attention_mask]
            attention_mask = torch.stack(attention_mask)
            labels = input_tokens["labels"]
            labels = [torch.tensor(label) for label in labels]
            labels = [pad_to_length(label, pad_length, -100, padding_side=padding_side) for label in labels]
            labels = torch.stack(labels)
        return BatchEncoding(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

import transformers
from abc import abstractmethod,ABC
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Dict

class VLProcessor(ABC):
    @abstractmethod
    def __init__(self,model_name_or_path,**kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def image_processor(self):
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self):# hack for some model uses unique processor, and we need to save it after training.
        raise NotImplementedError

    @abstractmethod
    def process_batch_conv(self,sources,system_message=None) -> dict: # tokenize a batch of conversations
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def format_prompt(prompt:str,img_paths:List[str]): #* add image placeholder or source to prompt. must be used in VLDPOTrainer.tokenize_row
        raise NotImplementedError

    @staticmethod
    def make_single_turn_conv(prompt:str,answer:str):
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
    
class LlavaProcessor(VLProcessor):
    def __init__(self,model_name_or_path,**kwargs) -> None:
        self.processor = transformers.LlavaProcessor.from_pretrained(model_name_or_path,**kwargs)
    
    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def image_processor(self):
        return self.processor.image_processor

    def save_pretrained(self):
        return self.processor.save_pretrained()

    def process_batch_conv(self,sources,system_message):
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
            "full": None,
            "raw_str": raw_texts,
        }
    @staticmethod
    def format_prompt(prompt:str,img_paths:List[str]):
        # currently not support multi images
        return "<image>\n"+prompt
    


class QwenVLProcessor(VLProcessor):
    def __init__(self,model_name_or_path,**kwargs) -> None:
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,**kwargs)
    
    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def image_processor(self):
        return None
    
    def save_pretrained(self):
        return None

    def process_batch_conv(self,sources:List[List[Dict]],system_message="You are a helpful assistant."):
        #FIXME: currently not support multi-turn conversation
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
                [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
            )
            _raw_text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                #* In generation mode, we only add this tokens
                _input_id = (
                    self.tokenizer(role).input_ids
                    + nl_tokens
                )
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
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID]
                        )
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
                        _target = (
                            [im_start]
                            + [IGNORE_TOKEN_ID]
                        )
                    answer_ids.append(_input_id[:])
                    answer_targets.append(_target[:])
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target)
            full_ids.append(input_id[:])
            full_targets.append(target[:])
            raw_texts.append(_raw_text)
            assert len(prompt_ids[-1]) == len(prompt_targets[-1])
            assert len(answer_ids[-1]) == len(answer_targets[-1])

        prompt_sequence_tokens = dict(
            input_ids=prompt_ids,
            labels=prompt_targets,
            attention_mask=[
                [id != self.tokenizer.pad_token_id for id in ids] for ids in prompt_ids
            ],
        )
        answer_sequence_tokens = dict(
            input_ids=answer_ids,
            labels=answer_targets,
            attention_mask=[
                [id != self.tokenizer.pad_token_id for id in ids] for ids in answer_ids
            ],
        )

        full_sequence_tokens = dict(
            input_ids=full_ids,
            labels=full_targets,
            attention_mask=[
                [id != self.tokenizer.pad_token_id for id in ids] for ids in full_ids
            ],
        )
        return {
            "prompt": prompt_sequence_tokens,
            "answer": answer_sequence_tokens,
            "full": full_sequence_tokens,
            "raw_str": raw_texts,
        }
    
    @staticmethod
    def format_prompt(prompt:str,img_paths:List[str]):
        out = []
        for i, img_path in enumerate(img_paths):
            out.append(f"Picture {i + 1}: <img>{img_path}</img>\n")
        out.append(prompt.strip())
        return "".join(out)
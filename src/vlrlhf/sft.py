from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
import transformers
import os
from vlrlhf.utils.auto_load import (
    MyAutoSFTTrainer,
    MyAutoProcessor,
    MyAutoSFTCollator,
    auto_load_rlmodel,
)
from vlrlhf.utils.common import safe_save_model_for_hf_trainer
from vlrlhf.utils.data import DATASET_MAP
from transformers.trainer_callback import TrainerCallback


# transformers.logging.set_verbosity_info()
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """

    data_path: Optional[str] = field(default=None, metadata={"help": "path of huggingface dataset cache dir"})
    data_ratio: Optional[float] = field(default=1.0, metadata={"help": "ratio of data to use"})
    image_root: Optional[str] = field(default=None, metadata={"help": "path of image root"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "name of the dataset"})
    model_name_or_path: Optional[str] = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the model name"})

    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    freeze_vision_tower: bool = field(default=True)
    merge_peft_model: bool = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = field(default=None)
    lora_bias: str = "none"
    q_lora: bool = False
    bits: int = 4
    modules_to_save: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.lora_target_modules is not None and self.lora_target_modules != "auto":
            self.lora_target_modules = self.lora_target_modules.split(",")
        if self.modules_to_save is not None:
            self.modules_to_save = self.modules_to_save.split(",")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = False
    use_flash_attention_2: bool = True
    dataset_num_proc: Optional[int] = field(
        default=4, metadata={"help": "number of processes to use for dataset loading"}
    )
    project_name: Optional[str] = field(default="VL-RLHF", metadata={"help": "wandb project name"})
    group_name: Optional[str] = field(default="Qwen-VL-Chat-sft", metadata={"help": "wandb group name"})
    resume_from_checkpoint: Optional[bool] = field(default=None)


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, LoraArguments))
    script_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_RUN_GROUP"] = training_args.group_name

    model, ref_model, lora_config = auto_load_rlmodel(script_args, training_args, lora_args)
    processor = MyAutoProcessor.from_pretrained(script_args.model_name_or_path)

    processor.train()

    dataset = DATASET_MAP[script_args.dataset_name](script_args)
    dataset_split = dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset_split["train"]
    train_dataset = train_dataset.select(range(int(len(train_dataset) * script_args.data_ratio)))
    eval_dataset = dataset_split["test"]

    collator = MyAutoSFTCollator(
        script_args.model_name_or_path,
        processor.tokenizer.pad_token_id,
        script_args.label_pad_token_id,
        processor=processor,
    )
    sft_trainer = MyAutoSFTTrainer(
        model_name_or_path=script_args.model_name_or_path,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        max_seq_length=script_args.max_length,
        peft_config=lora_config,
        dataset_num_proc=training_args.dataset_num_proc,
    )
    if training_args.use_lora:
        sft_trainer.add_callback(PeftSavingCallback())
    sft_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    sft_trainer.save_state()
    safe_save_model_for_hf_trainer(sft_trainer, training_args.output_dir, lora_args)
    processor.save_pretrained(training_args.output_dir)
    if script_args.merge_peft_model and training_args.use_lora:
        merged_model = model.merge_peft_model()
        merged_dir = os.path.join(training_args.output_dir, "merged")
        merged_model.save_pretrained(merged_dir)

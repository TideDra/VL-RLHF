from dataclasses import dataclass, field
from typing import Optional
from vlrlhf.utils.data import DATASET_MAP
from transformers import HfArgumentParser

import transformers
import os
from vlrlhf.utils.auto_load import MyAutoDPOCollator, MyAutoDPOTrainer, MyAutoProcessor, auto_load_rlmodel
from vlrlhf.utils.common import safe_save_model_for_hf_trainer

from transformers.trainer_callback import TrainerCallback


# transformers.logging.set_verbosity_info()
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    score_margin: Optional[float] = field(
        default=-1,
        metadata={
            "help": """Minimal score gap between chosen and rejected responses if score is given in the dataset.
            Set to -1 to select the pair with the largest score gap for each prompt."""
        },
    )
    # training parameters
    data_path: Optional[str] = field(default=None, metadata={"help": "path of huggingface dataset cache dir"})
    data_ratio: Optional[float] = field(default=1.0, metadata={"help": "ratio of data to use"})
    image_root: Optional[str] = field(default=None, metadata={"help": "path of image root"})
    dataset_name: Optional[str] = field(default="vlfeedback_paired", metadata={"help": "name of the dataset"})
    model_name_or_path: Optional[str] = field(default="llava-hf/llava-1.5-7b-hf", metadata={"help": "the model name"})

    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"},
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    freeze_vision_tower: bool = field(default=True)
    loss_type: str = field(default="sigmoid")


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
        default=16, metadata={"help": "number of processes to use for dataset loading"}
    )
    project_name: Optional[str] = field(default="VL-RLHF", metadata={"help": "wandb project name"})
    group_name: Optional[str] = field(default="llava-1.5-7b-dpo", metadata={"help": "wandb group name"})
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
    data_collator = MyAutoDPOCollator(
        script_args.model_name_or_path,
        pad_token_id=processor.tokenizer.pad_token_id,
        label_pad_token_id=script_args.label_pad_token_id,
        is_encoder_decoder=model.config.is_encoder_decoder,
        processor=processor,
    )

    dpo_trainer = MyAutoDPOTrainer(
        script_args.model_name_or_path,
        model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        label_pad_token_id=script_args.label_pad_token_id,
        data_collator=data_collator,
        peft_config=lora_config,
        loss_type=script_args.loss_type,
        ref_model=ref_model,
        dataset_num_proc=training_args.dataset_num_proc,
    )
    if training_args.use_lora:
        dpo_trainer.add_callback(PeftSavingCallback())
    dpo_trainer.use_dpo_data_collator = True

    dpo_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    dpo_trainer.save_state()
    safe_save_model_for_hf_trainer(dpo_trainer, training_args.output_dir, lora_args)
    processor.save_pretrained(training_args.output_dir)

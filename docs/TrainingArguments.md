# Arguments Overview
This guide lists the arguments you can adjust in the training scripts.

Since VL-RLHF is built on [Transformers](https://github.com/huggingface/transformers), the arguments of Transformers are also available for VL-RLHF Trainer. So, here we only list the arguments added by VL-RLHF.

## Common
These are arguments shared by all types of VL-RLHF Trainers:
- `--model_name_or_path`: Path of the pretrained model weights.
- `--max_length`: Max length of each sample.
- `--max_prompt_length`: Max length of the prompt.
- `--max_target_length`: Max length of the target text.
- `--dataset_name`: The name of the dataset. can be `vlfeedback_paired` for the [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback) dataset, `rlhfv` for the [RLHF-V](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) dataset, `vlquery_json` for customized multimodal conversation data stored in json format, `plain_dpo` for customized multimodal comparison data stored in json format.
- `--data_path`: Path to the json file. **Only needed for customized dataset**. If you use VLFeedback or RLHF-V, these datasets will be automatically downloaded from huggingface and loaded via the `datasets` package.
- `--image_root`: Root directory of the images. **Only needed for customized dataset**. It will be joined with the image path of each sample in the json file.
- `--data_ratio`: Ratio between the number of training data and evaluation data.
- `--dataset_num_proc`: Number of processors for processing data.
- `--freeze_vision_tower`: Whether to freeze the vision encoder of the model. Defaults to `True`.
- `--lora_r`: LoRA rank.
- `--lora_alpha`: LoRA alpha.
- `--lora_dropout`: LoRA dropout.
- `--lora_target_modules`: LoRA target modules. Split by `,`, e.g. `"c_attn,attn.c_proj,w1,w2"`. You can set it to `auto` to use default lora target modules.
- `--lora_bias`: LoRA bias.
- `--use_lora`: Whether to use LoRA. Defaults to `False`
- `--q_lora`: Whether to use QLoRA. Defaults to `False`.
- `--bits`: Bits of QLoRA.
- `--modules_to_save`: Additional modules that should be saved in the checkpoint. Split by `,`.
- `--use_flash_attention_2`: Whether to use FlashAttention2 for effective training.
- `--project_name`: Name of the project. Used by wandb.
- `--group_name`: Group name of this experiment. Used by wandb.

## DPO
- `--beta`: beta in DPO loss.
- `--score_margin`: Currently only used for VLFeedback dataset. For a pair of responses, only when the difference of their scores is larger than `score_margin` can they be selected as a training sample. Defaults to `-1`, which uses all pairs.
- `--loss_type`: Same as the `loss_type` argument of TRL DPOTrainer. Can be one of `["sigmoid", "hinge", "ipo", "kto_pair", "ddpo"]`.

## DDPO
DDPO is a variant of DPO, where the `loss_type` is set to `ddpo`.

## KTO (paired)
KTO (paired) is a variant of DPO, where the `loss_type` is set to `kto_pair`.

## SFT
There is currently no additional arguments for SFT.

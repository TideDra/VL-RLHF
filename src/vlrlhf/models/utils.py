from dataclasses import dataclass
from ..base import (
    VLDPODataCollatorWithPadding,
    VLDPOTrainer,
    VLModelWithValueHead,
    VLPPODataCollator,
    VLPPOTrainer,
    VLProcessor,
    VLRewardModel,
    VLRMDataCollatorWithPadding,
    VLRMTrainer,
    VLSFTDataCollatorWithPadding,
    VLSFTTrainer,
)
from transformers import PreTrainedModel


@dataclass
class ModelCoreMapper:
    model: PreTrainedModel
    processor: VLProcessor
    dpo_collator: VLDPODataCollatorWithPadding
    dpo_trainer: VLDPOTrainer
    reward_model: VLRewardModel
    value_model: VLModelWithValueHead
    reward_collator: VLRMDataCollatorWithPadding
    reward_trainer: VLRMTrainer
    sft_collator: VLSFTDataCollatorWithPadding
    sft_trainer: VLSFTTrainer
    ppo_collator: VLPPODataCollator
    ppo_trainer: VLPPOTrainer

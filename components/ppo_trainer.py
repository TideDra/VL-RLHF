from typing import Optional, Union, Any, Callable, List
from datasets import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from components.model import VLModelWithValueHead, VLRewardModel
from trl import PPOTrainer, PPOConfig
from components.processor import VLProcessor
from tqdm import tqdm
from abc import ABC
import torch
import wandb
from loguru import logger
from accelerate.utils import gather_object
class VLPPOTrainer(PPOTrainer,ABC):
    def __init__(
        self,
        config: PPOConfig = None,
        model: VLModelWithValueHead = None,
        ref_model: VLModelWithValueHead | None = None,
        reward_model: VLRewardModel | None = None,
        processor: VLProcessor | None = None,
        dataset: Any | Dataset | None = None,
        optimizer: Optimizer | None = None,
        data_collator: Optional[Callable] = None,
        num_shared_layers: int | None = None,
        lr_scheduler: _LRScheduler | None = None,
        generation_kwargs: dict = {}
    ):
        if getattr(config, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model.gradient_checkpointing_enable(config.gradient_checkpointing_kwargs)
        self.processor = processor
        self.generation_kwargs = generation_kwargs
        if model.supports_rm_adapter and reward_model is not None:
            raise ValueError("Model supports reward model adapter, but you also give a reward model. Please only use one of them to compute rewards.")
        self.reward_model = reward_model
        dataset = dataset.map(self.tokenize_row)
        super().__init__(
            config,
            model,
            ref_model,
            processor.tokenizer,
            dataset,
            optimizer,
            data_collator,
            num_shared_layers,
            lr_scheduler,
        )
        self.is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )
        if self.is_deepspeed_used:
            # without this, the program will stuck at pad_across_processes in zero2/3
            self.generation_kwargs["synced_gpus"] = True
    def tokenize_row(self,element):
        query = element["query"]
        image = element["image"]
        query = self.processor.format_multimodal_prompt(query,image)
        conv = self.processor.make_single_turn_conv(query,'')

        tokens = self.processor.process_batch_conv([conv])["full"]
        tokens = {
            k: v[0]
            for k, v in tokens.items() if k != "labels" # we don't need labels
        }
        tokens["query"] = query
        tokens['image'] = image
        return tokens
    
    def train(self):
        if self.accelerator.is_main_process:
            bar = tqdm(total=len(self.dataloader),desc="Launching...")
        for epoch, batch in enumerate(self.dataloader):
            query_tensors = batch["input_ids"]
            #### Get response from SFTModel
            if self.accelerator.is_main_process:
                bar.set_description("generating response")
            with torch.no_grad():
                response_tensors = self.generate(query_tensors, return_prompt=False,batch_size=32,**self.generation_kwargs)

            batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            #### Compute reward score
            convs = [self.processor.make_single_turn_conv(q,r) for q, r in zip(batch["query"], batch["response"])]
            inputs = self.processor.process_batch_conv(convs)['full']
            inputs.pop("labels")
            padding_side_default = self.tokenizer.padding_side
            if not self.is_encoder_decoder:
                self.tokenizer.padding_side = "left"
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                return_tensors="pt",
            ).to(self.current_device)
            self.tokenizer.padding_side = padding_side_default
            
            if self.accelerator.is_main_process:
                bar.set_description_str("computing reward score")
            skip_batch = False
            try:
                if self.reward_model is not None:
                    with torch.no_grad():
                        rewards = self.reward_model(**padded_inputs)[0]
                else:
                    rewards = self.accelerator.unwrap_model(self.model).compute_reward_score(**padded_inputs) #compute_reward_score sets no_grad
            except Exception as e:
                logger.warning("Error when computing reward score. Skip this batch. See the following exception for more details.")
                logger.exception(e)
                print(batch["response"])
                skip_batch = True
            skip_batch = [skip_batch]
            gather_object(skip_batch)
            if torch.tensor(skip_batch).any():
                print(skip_batch)
                continue
            # self.step needs a list of rewards, then it turn the list into a tensor again. This is really stupid.
            rewards = [reward for reward in rewards]
            #### Run PPO step
            # we don't pass response_mask because self.generate has already removed padding
            if self.accelerator.is_main_process:
                bar.set_description_str("running ppo step")
            stats = self.step(query_tensors, response_tensors, rewards)
            rewards = torch.tensor(rewards,dtype=torch.float,device=self.current_device) # bf16 cannot be turned to numpy for logging
            batch = {k: v[:self.config.per_device_gamelog_size] for k, v in batch.items()}
            rewards = rewards[:self.config.per_device_gamelog_size]
            batch['image'] = [wandb.Image(image) for image in batch['image']]
            batch['query'] = [self.processor.remove_image_placeholder(query) for query in batch['query']]
            self.log_stats(stats, batch, rewards,columns_to_log=['image','query','response'])
            if self.accelerator.is_main_process:
                bar.update()

class LlavaPPOTrainer(VLPPOTrainer):
    ...

class QwenVLPPOTrainer(VLPPOTrainer):
    ...
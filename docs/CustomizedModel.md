# Customized Model
To add your own model to VL-RLHF framework, you need to create a directory in `src/vlrlhf/models`, and implement all the APIs we need in `__init__.py`. You can refer to other models that we have implemented in `src/vlrlhf/models`.

## `__init__.py`
Here we list the APIs you need to implement in `__init__.py`. Most of them are inherited from our predefined abstract class, and there is no additional code needed if you have no special requirements.

- Model class: You should implement the following methods or property in the model class.
    - `default_lora_target`: A class property which is a list that contains default LoRA target modules. When `--lora_target_modules` is set to `auto` in the training command, this `default_lora_target` is used.
    - `get_vision_tower`: A class method that return the vision encoder.
    - `freeze_vision_tower`: A class method that freeze vision encoder. When `--freeze_vision_tower` is set to `True` in the training command, this method is used to freeze vision encoder.
    - `prepare_default_generation_kwargs`: A class method that return default generation kwargs dict, which is used as the generation config during evaluation.
- Processor: A subclass of `vlrlhf.base.processor.VLProcessor`. You should implement the following abstract method:
    - `__init__`: The initialization method.
    - `tokenizer`: The class property which returns tokenizer.
    - `chat_template`: The class property which defines the chat template.
    - `image_processor`: The class property which returns the image processor.
    - `save_pretrained`: The class method which will be called after training to save the processor.
    - `process_batch_conv`: The class method which tokenizes a batch of conversations
    - `format_multimodal_prompt`: The class method which adds image placeholders to the raw prompt.
    - `remove_image_placeholder`: The class method which removes the image placeholders in given prompt.
    - `is_multimodal_prompt_valid`: The class method which checks whether the given prompt contains the image placeholder.
    - `train`: The class method which turns on training mode, e.g. setting the tokenizer to right-padding mode. It will be called before training.
    - `infer`: The class method which turns on inference mode, e.g. setting the tokenizer to left-padding mode. It will be called before evaluation.
    - `__call__`: The call method. The abstract class has implemented most of features of this method, which is able to automatically tokenize the text. What you need to do is to call it via `super().__call__` and then process images manually in your implementation.

- DataCollator: You need to implement a DataCollator for each algorithm. We have implemented abstract classes for all of them in `vlrlhf.base.collator`. What you need to do is to create a subclass of them and process the images manually, like this:
```python
class LlavaDPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["img_input_dict"] = dict(
            pixel_values=self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        )
        return padded_batch
```
where `padded_batch["img_input_dict"]` is a dict contains all the inputs related to image (or other inputs that are not processed in `super().__call__`)

- Trainer: You need to implement a Trainer for each algorithm. We have implemented abstract classes for all of them in `vlrlhf.base.trainer`. What you need to do is to create an empty subclass like:
```python
class LlavaSFTTRainer(VLSFTTrainer):
    ...
```
The abstract class has implemented most of features for training. If you have any special requirement, just overwrite the related class methods.

- core_mapper: Don't forget to map all the classes to the corresponding attributes of the variable `core_mapper`:
```python
core_mapper = ModelCoreMapper(
    model=LlavaForRL,
    processor=LlavaProcessor,
    dpo_collator=LlavaDPODataCollatorWithPadding,
    dpo_trainer=LlavaDPOTrainer,
    reward_model=LlavaRewardModel,
    value_model=LlavaWithValueHead,
    reward_collator=LlavaRMDataCollatorWithPadding,
    reward_trainer=LlavaRMTrainer,
    sft_collator=LlavaSFTDataCollatorWithPadding,
    sft_trainer=LlavaSFTTRainer,
    ppo_collator=LlavaPPODataCollator,
    ppo_trainer=LlavaPPOTrainer,
)
```
VL-RLHF imports all the components via `core_mapper`.

## `auto_load.py`
You also need to add some configuration in `src/vlrlhf/utils/auto_load.py`, so that we can map a model checkpoint to the corresponding model class.

You can find a variable `MODEL_NICKNAME_MAP` in the file. Just add an item to it, like:
```python
MODEL_NICKNAME_MAP = {
    ...
    "LlavaForConditionalGeneration": "Llava",
}
```
where the key `LlavaForConditionalGeneration` is the class name specified in the model checkpoint, and the value `LLaVA` is the **name of the directory** that contains the above `__init__.py` file.

If your model supports FlashAttention2, also add it to `FLASH_ATTN_MODELS`, like:
```python
FLASH_ATTN_MODELS = [
    ...
    "LlavaForConditionalGeneration",
]
```

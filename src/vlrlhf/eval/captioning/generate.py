import json
from PIL import Image
from vlrlhf.utils.auto_load import MyAutoModel, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset,DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
import random
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/gozhang/data_dir/coco2017/test2017")
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


class CaptioningDataset(Dataset):
    def __init__(self,data_root,sample_num) -> None:
        super().__init__()
        self.data_root = data_root
        self.images = os.listdir(data_root)
        self.images = random.sample(self.images,sample_num)
        self.prompt = "Describe the picture in detail."
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            'image':os.path.join(self.data_root,self.images[index]),
            'question':self.prompt
        }

def collator(batch):
    questions = [b['question'] for b in batch]
    images = [b['image'] for b in batch]
    prompt = [processor.format_multimodal_prompt(q,img) for q,img in zip(questions,images)]
    inputs = processor(texts=prompt,images_path=images,padding_side='left',check_format=False)
    return images,inputs


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    sample_num = args.sample_num
    model_path = args.model_path
    model = MyAutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16)
    if isinstance(model,PeftModel):
        model_path = model.peft_config['default'].base_model_name_or_path
    if args.processor_path is None:
        args.processor_path = model_path
    processor = MyAutoProcessor.from_pretrained(args.processor_path)
    processor.infer()
    tokenizer = processor.tokenizer

    model.to('cuda')
    model.to(torch.bfloat16)
    dataset = CaptioningDataset(data_root,sample_num)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(model_path)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for images,inputs in dataloader:
            inputs.to('cuda')
            outputs = model.generate(**inputs,generation_config=generation_config,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            results.extend([{'image':img,'caption':res} for img,res in zip(images,responses)])
            bar.update(len(responses))
    with open(args.output_path,'w') as f:
        json.dump(results,f,indent=4)
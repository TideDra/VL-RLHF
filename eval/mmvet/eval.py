import json
from PIL import Image
from utils.auto_load import MyAutoModel, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset,DataLoader
import torch
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/gozhang/data_dir/mm-vet")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default="mmvet_result.json")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


class MMVetDataset(Dataset):
    def __init__(self,data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, "mm-vet.json"), "r") as f:
            data = json.load(f)
        self.data = [(k,v) for k,v in data.items()]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'id':self.data[index][0],
                'image':os.path.join(self.data_root,'images',self.data[index][1]['imagename']),
                'question':self.data[index][1]['question']}

def collator(batch):
    ids = [b['id'] for b in batch]
    questions = [b['question'] for b in batch]
    images = [b['image'] for b in batch]
    if processor.__class__.__name__ == 'QwenVLProcessor':
        # Without this prefix, the model works better.
        prompt = [processor.format_multimodal_prompt(q,img).replace('Picture 1: ','') for q,img in zip(questions,images)]
    else:
        prompt = [processor.format_multimodal_prompt(q,img) for q,img in zip(questions,images)]
    inputs = processor(texts=prompt,images_path=images,padding_side='left',check_format=False)
    return ids,inputs


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = MyAutoProcessor.from_pretrained(args.model_path)
    processor.infer()
    tokenizer = processor.tokenizer

    model = MyAutoModel.from_pretrained(args.model_path,torch_dtype=torch.bfloat16)
    model.to('cuda')
    dataset = MMVetDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(args.model_path)
    results = {}
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for ids, inputs in dataloader:
            inputs.to('cuda')
            outputs = model.generate(**inputs,generation_config=generation_config,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id,res in zip(ids,responses):
                results[id]=res
            bar.update(len(responses))
    with open(args.output_path,'w') as f:
        json.dump(results,f,indent=4)
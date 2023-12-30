import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration,AutoTokenizer
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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
                'question':"USER: <image>\n"+self.data[index][1]['question']+" ASSISTANT:"}

def collator(batch):
    ids = [b['id'] for b in batch]
    questions = [b['question'] for b in batch]
    images = [Image.open(b['image']) for b in batch]
    inputs = processor(text=questions,images=images,return_tensors="pt",padding=True)
    return ids,inputs



if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor.tokenizer.pad_token = processor.tokenizer.bos_token
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16)
    model.to('cuda')
    dataset = MMVetDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    results = {}
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for ids, inputs in dataloader:
            inputs.to('cuda')
            inputs['pixel_values'] = inputs['pixel_values'].half()
            outputs = model.generate(**inputs,temperature=args.temperature,do_sample=True,max_new_tokens=args.max_new_tokens,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id,res in zip(ids,responses):
                results[id]=res
            bar.update(len(responses))
    with open(args.output_path,'w') as f:
        json.dump(results,f,indent=4)
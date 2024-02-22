import json
from PIL import Image
from vlrlhf.utils.auto_load import MyAutoModel, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset,DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import pandas as pd
from peft import PeftModel
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/gozhang/data_dir/mm-vet")
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--processor_path", type=str, default=None)
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
                'question':self.data[index][1]['question'],
                'answer':self.data[index][1]['answer'],
                'category':','.join(self.data[index][1]['capability'])}

def collator(batch):
    ids = [b['id'] for b in batch]
    categories = [b['category'] for b in batch]
    answers = [b['answer'] for b in batch]
    questions = [b['question'] for b in batch]
    images = [b['image'] for b in batch]
    prompt = [processor.format_multimodal_prompt(q,img).replace('Picture 1: ','') for q,img in zip(questions,images)]
    inputs = processor(texts=prompt,images_path=images,padding_side='left',check_format=False)
    return ids,answers,questions,categories,inputs


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
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
    dataset = MMVetDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(model_path)
    results = defaultdict(list)
    json_results = {}
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for ids, answers,questions,categories,inputs in dataloader:
            inputs.to('cuda')
            outputs = model.generate(**inputs,generation_config=generation_config,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id,answer,question,category,res in zip(ids,answers,questions,categories,responses):
                results['index'].append(id)
                results['answer'].append(answer)
                results['question'].append(question)
                results['prediction'].append(res)
                results['category'].append(category)
                json_results[id] = res
            bar.update(len(responses))
    #pd.DataFrame(results).to_excel('mmvet_result.xlsx')
    with open(args.output_path,'w') as f:
        json.dump(dict(item for item in zip(results['index'],results['prediction'])),f,indent=4)
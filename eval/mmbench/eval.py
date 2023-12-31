from PIL import Image
from utils.auto_load import MyAutoModel, MyAutoProcessor, MyAutoGenerationConfig
from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
import argparse
import base64
import pandas as pd
import io
from collections import defaultdict
import tempfile
from time import time
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default="mmbench_result.xlsx")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()



class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        self.temp_img_dir = tempfile.TemporaryDirectory()

    def dump_image_to_tempfile(self,base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        name = os.path.join(self.temp_img_dir.name, f'{time()}.jpg')
        image.save(name)
        return name
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = self.dump_image_to_tempfile(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']
        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        if data['context'] is not None:
            prompt = data['context'] + ' ' + data['question'] + ' ' + data['options']+ '\n' + 'please only output the option letter.'
        else:
            prompt = data['question'] + ' ' + data['options']+ '\n' + 'please only output the option letter.'

        data['prompt'] = prompt
        return data
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

def collator(batch):
    concat_batch = defaultdict(list)
    for data in batch:
        for key, item in data.items():
            concat_batch[key].append(item)
    concat_batch['prompt'] = [processor.format_multimodal_prompt(promt,img_path) for promt,img_path in zip(concat_batch['prompt'],concat_batch['img'])]
    concat_batch['input'] = processor(texts=concat_batch['prompt'],images_path=concat_batch['img'],padding_side='left')
    return concat_batch


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = MyAutoProcessor.from_pretrained(args.model_path)
    generation_config = MyAutoGenerationConfig.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    processor.infer()
    model = MyAutoModel.from_pretrained(args.model_path,torch_dtype=torch.bfloat16)
    model.to('cuda')
    dataset = MMBenchDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    results = defaultdict(list)
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch['input']
            inputs.to('cuda')
            outputs = model.generate(**inputs,use_cache=True,generation_config=generation_config)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for k,v in batch.items():
                if k in ['question','options_dict','category','l2-category','index']:
                    results[k].extend(v)
            results['response'].extend(responses)
            bar.update(len(responses))
    dataset.temp_img_dir.cleanup()
    answer_upload = defaultdict(list)

    for question,options_dict,category,l2_category,index,response in zip(results['question'],results['options_dict'],results['category'],results['l2-category'],results['index'],results['response']):
        choice_A = options_dict.get('A',"")
        choice_B = options_dict.get('B',"")
        choice_C = options_dict.get('C',"")
        choice_D = options_dict.get('D',"")
        split = 'dev'
        answer_upload['question'].append(question)
        answer_upload['A'].append(choice_A)
        answer_upload['B'].append(choice_B)
        answer_upload['C'].append(choice_C)
        answer_upload['D'].append(choice_D)
        answer_upload['prediction'].append(response)
        answer_upload['category'].append(category)
        answer_upload['l2_category'].append(l2_category)
        answer_upload['index'].append(index)
        answer_upload['split'].append(split)

    answer_upload = pd.DataFrame(answer_upload)
    answer_upload.to_excel(args.output_path,index=False)
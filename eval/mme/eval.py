import json
from PIL import Image
from utils.auto_load import MyAutoProcessor, MyAutoProcessor, MyAutoModel, MyAutoGenerationConfig
from torch.utils.data import Dataset,DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import jsonlines
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/mnt/gozhang/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default="mme_result.json")
    parser.add_argument("--batch_size", type=int, default=16)


    return parser.parse_args()


class MMEDataset(Dataset):
    def __init__(self,data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with jsonlines.open(os.path.join(data_root, "mme.jsonl"), "r") as f:
            self.data = [item for item in f]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'question_id':self.data[index]["question_id"],
                'image':os.path.join(self.data_root,self.data[index]['image']),
                "text":self.data[index]['text'],
                'prompt':self.data[index]['text']}

def collator(batch):
    question_ids = [b['question_id'] for b in batch]
    prompts = [b['prompt'] for b in batch]
    texts = [b['text'] for b in batch]
    img_path = [b['image'] for b in batch]
    prompts = [processor.format_multimodal_prompt(prompt,img) for prompt,img in zip(prompts,img_path)]
    inputs = processor(texts=prompts,images_path=img_path,padding_side='left')
    return question_ids,texts,inputs



if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = MyAutoProcessor.from_pretrained(args.model_path)
    processor.infer()
    tokenizer = processor.tokenizer
    model = MyAutoModel.from_pretrained(args.model_path,torch_dtype=torch.float16)
    model.to('cuda')
    dataset = MMEDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    generation_config = MyAutoGenerationConfig.from_pretrained(args.model_path)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for question_ids,texts,inputs in dataloader:
            inputs.to('cuda')
            outputs = model.generate(**inputs,generation_config=generation_config,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for question_id,text,res in zip(question_ids,texts,responses):
                results.append({"question_id":question_id,"prompt":text,"text":res})
            bar.update(len(responses))
    with jsonlines.open(args.output_path,"w") as f:
        f.write_all(results)
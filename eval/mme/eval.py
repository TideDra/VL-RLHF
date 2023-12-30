import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration,AutoTokenizer
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
    #parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    return parser.parse_args()


class MMEDataset(Dataset):
    def __init__(self,data_root) -> None:
        super().__init__()
        self.data_root = data_root
        with jsonlines.open(os.path.join(data_root, "llava_mme.jsonl"), "r") as f:
            self.data = [item for item in f]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'question_id':self.data[index]["question_id"],
                'image':os.path.join(self.data_root,self.data[index]['image']),
                "text":self.data[index]['text'],
                'prompt':"USER: <image>\n"+self.data[index]['text']+" ASSISTANT:"}

def collator(batch):
    question_ids = [b['question_id'] for b in batch]
    prompts = [b['prompt'] for b in batch]
    texts = [b['text'] for b in batch]
    images = [Image.open(b['image']) for b in batch]
    inputs = processor(text=prompts,images=images,return_tensors="pt",padding=True)
    return question_ids,texts,inputs



if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor.tokenizer.pad_token = processor.tokenizer.bos_token
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16)
    model.to('cuda')
    dataset = MMEDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    results = []
    bar = tqdm(total=len(dataset))
    model.eval()
    with torch.inference_mode():
        for question_ids,texts,inputs in dataloader:
            inputs.to('cuda')
            inputs['pixel_values'] = inputs['pixel_values'].half()
            outputs = model.generate(**inputs,do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True)
            input_token_len = inputs['input_ids'].shape[1]
            responses=tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for question_id,text,res in zip(question_ids,texts,responses):
                results.append({"question_id":question_id,"prompt":text,"text":res})
            bar.update(len(responses))
    with jsonlines.open(args.output_path,"w") as f:
        f.write_all(results)
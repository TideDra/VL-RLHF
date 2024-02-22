import json
from torch.utils.data import Dataset,DataLoader

import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import pandas as pd

from sglang import function,  user, assistant, gen, set_default_backend, RuntimeEndpoint,image
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/gozhang/data_dir/mm-vet")
    parser.add_argument("--endpoint", type=str, default="http://localhost:30000")
    parser.add_argument("--output_path", type=str, default="llava1.5-34b-sgl.json")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

@function
def image_qa(s, image_path, question):
    s += user(image(image_path) + question)
    s += assistant(gen("answer"))

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
    inputs = [{'image_path':i,'question':'<image>\n'+q} for i,q in zip(images,questions)]
    return ids,answers,questions,categories,inputs


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    set_default_backend(RuntimeEndpoint(args.endpoint))
    dataset = MMVetDataset(data_root)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)

    results = defaultdict(list)
    json_results = {}
    bar = tqdm(total=len(dataset))

    for ids, answers,questions,categories,inputs in dataloader:
        states = image_qa.run_batch(inputs,max_new_tokens=1024,temperature=0)
        responses = [s['answer'] for s in states]
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
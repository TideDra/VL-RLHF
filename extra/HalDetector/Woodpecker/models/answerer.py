import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Dict
from utils.auto_load import MyAutoModel,MyAutoProcessor, MyAutoGenerationConfig


def get_answer_or_prepare(processor, model, raw_img_path,img_path, qs,batch):
    if batch.get((raw_img_path,img_path,qs), None) is None:
        if img_path is not None:
            new_qs = f"Picture 2 is cropped from Picture 1. Based on the global context provided by Picture 1 and focused on Picture 2, {qs}"
            prompt = processor.format_multimodal_prompt(prompt=new_qs, img_paths=[raw_img_path,img_path])
        else:
            prompt = processor.format_multimodal_prompt(prompt=qs, img_paths=raw_img_path)
        batch[(raw_img_path,img_path,qs)] = {'prompt': prompt, 'img_path':img_path}
        return None
    else:
        return batch[(raw_img_path,img_path,qs)].get('output', None)

def process_batch(processor, model,generation_config,batch):
    prompts = []
    img_paths = []
    for v in batch.values():
        prompts.append(v['prompt'])
        img_paths.append(v['img_path'])

    inputs = processor(texts=prompts, images_path=img_paths, padding_side='left',check_format=False)
    inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs,use_cache=True,generation_config=generation_config)
    input_token_len = inputs['input_ids'].shape[1]
    generated_text = processor.tokenizer.batch_decode(generated_ids[:,input_token_len:], skip_special_tokens=True)
    for idx, k in enumerate(batch.keys()):
        batch[k]['output'] = generated_text[idx].strip()


def get_all_answers(processor, model, entity_list, qs, ent_info, input_img_path, cur_answers,batch):
    # This should return a dict. Since a question may correspond to multiple instances of a same kind of object.
    # case 1: involve multiple entities or 'where' type question: use the whole img.
    if len(entity_list)>1 or 'where' in qs.lower() or any([ent not in ent_info for ent in entity_list]):

        answer = get_answer_or_prepare(processor, model, input_img_path,None,qs,batch) 
        cur_answers.setdefault('overall', [])   # use a special category 'overall' to denote answers that involve multiple objects.
        cur_answers['overall'].append((qs, answer))
    else:
        entity = entity_list[0]
        # case 2: single entity : single/multiple instances.
        for idx, img_path in enumerate(ent_info[entity]['crop_path']):

            answer = get_answer_or_prepare(processor, model, input_img_path,img_path, qs,batch)
            cur_answers.setdefault(entity, [])
            if idx + 1 > len(cur_answers[entity]):
                cur_answers[entity].append([])
            cur_answers[entity][idx].append((qs, answer))
    return cur_answers

class Answerer:
    '''
        Input: 
            'generated_questions': a list of 2-ele list, each [qs(str), involved entities(str)]
            'entity_info': A dict recording the global object information.
            key: obj name. (obj1 | obj2 | obj3)
            value:
                {
                    total_count: detected counts of that obj.
                    
                    crop_path: a list of str, denoting the path to cached intermediate file, i.e., cropped out region of that obj.
                        Note: if total_count > 1, may use the whole image in the following steps.
                        
                    bbox: each [x1, y1, x2, y2], normalized coordinates of left-top and right-bottom corners of bounding boxes.
                }
        Output:
            'generated_answers': An 1-d list of dict. Each dict in the list contains all the (qs, ans) tuple for each object instance.
                                {
                                    overall: [(qs, answer), ...]
                                    entity:  [
                                                [(qs, answer), ...]   (for instance 1 of this type of entity)
                                                    ...
                                             ]
                                }
    '''
    
    def __init__(self, val_model_path,device='cuda'):
        self.device = device
        self.processor = MyAutoProcessor.from_pretrained(val_model_path)
        self.processor.infer()
        self.model = MyAutoModel.from_pretrained(val_model_path, torch_dtype=torch.bfloat16).to(device)
        self.generation_config = MyAutoGenerationConfig.from_pretrained(val_model_path)

    def generate_answers(self, sample: Dict):
        generated_qs = sample['generated_questions']
        global_entity_dict = sample['entity_info']
        # prepare batch
        batch = {}
        all_answers = []
        for gen_qs in generated_qs:
            # border case: no question asked.
            if len(gen_qs) == 0:
                all_answers.append({})
                continue
            cur_answers = {}
            for cur_qs in gen_qs:
                qs, entity = cur_qs # qs is a str. entity is also a str. may contain multiple entity connected by periods.
                entity_list = entity.split('.')
                entity_list = [e.strip() for e in entity_list if e.strip()]
                
                cur_answers = get_all_answers(self.processor, self.model, entity_list, qs, global_entity_dict, sample['img_path'], cur_answers,batch)
            all_answers.append(cur_answers)

        process_batch(self.processor, self.model,self.generation_config,batch)
        all_answers = []
        for gen_qs in generated_qs:
            # border case: no question asked.
            if len(gen_qs) == 0:
                all_answers.append({})
                continue
            cur_answers = {}
            for cur_qs in gen_qs:
                qs, entity = cur_qs # qs is a str. entity is also a str. may contain multiple entity connected by periods.
                entity_list = entity.split('.')
                entity_list = [e.strip() for e in entity_list if e.strip()]
                
                cur_answers = get_all_answers(self.processor, self.model, entity_list, qs, global_entity_dict, sample['img_path'], cur_answers,batch)
            all_answers.append(cur_answers)

        sample['generated_answers'] = all_answers
        return sample
    
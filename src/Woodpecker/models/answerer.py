import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
import torch
from PIL import Image

from typing import Dict, List
from .utils import image_qa
from sglang.utils import http_request
def get_answer_or_prepare(raw_img_path,img_path, qs,batch):
    if batch.get((raw_img_path,img_path,qs), None) is None:
        if img_path is not None:
            new_qs = f"Focus on the region annotated by the red bounding box in this image, {qs} Give me a short answer."
            prompt = new_qs
            batch[(raw_img_path,img_path,qs)] = {'prompt': prompt, 'img_path':img_path}
        else:
            prompt = qs
            batch[(raw_img_path,img_path,qs)] = {'prompt': prompt, 'img_path':raw_img_path}
        return None
    else:
        return batch[(raw_img_path,img_path,qs)].get('output', None)

def process_batch(batch,endpoint):
    minibatch_size=32
    minibatch = []
    for idx, k in enumerate(batch.keys()):
        minibatch.append({"image_path":batch[k]['img_path'],"question":batch[k]['prompt'],'k':k})
        if len(minibatch) == minibatch_size or idx == len(batch)-1:
            #http_request(endpoint.base_url+"/flush_cache")
            states = image_qa.run_batch(
                [{"image_path":v['image_path'],"question":v['question']} for v in minibatch],
                temperature=0,
                max_new_tokens=64,
                backend=endpoint
                )
            for sample,s in zip(minibatch,states):
                batch[sample['k']]['output'] = s['answer']
            minibatch = []


def get_all_answers(entity_list, qs, ent_info, input_img_path, cur_answers,batch,is_multi):
    # This should return a dict. Since a question may correspond to multiple instances of a same kind of object.
    # case 1: involve multiple entities or 'where' type question: use the whole img.
    overall = len(entity_list)>1 or \
            'where' in qs.lower() or \
            any([ent not in ent_info for ent in entity_list]) or \
            ent_info[entity_list[0]]['total_count'] == "unknown" or \
            is_multi
    if overall:
        answer = get_answer_or_prepare(input_img_path,None,qs,batch) 
        cur_answers.setdefault('overall', [])   # use a special category 'overall' to denote answers that involve multiple objects.
        cur_answers['overall'].append((qs, answer))
    else:
        entity = entity_list[0]
        # case 2: single entity : single/multiple instances.
        for idx, img_path in enumerate(ent_info[entity]['crop_path']):

            answer = get_answer_or_prepare(input_img_path,img_path, qs,batch)
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
    
    def __init__(self,endpoint):
        self.endpoint = endpoint

    def generate_batch_answers(self, samples: List[Dict]):
        batch = {}
        # prepare batch
        for idx,sample in enumerate(samples):
            samples[idx] = self.generate_answers(sample,batch)
        process_batch(batch,self.endpoint)
        # assign value
        for idx,sample in enumerate(samples):
            samples[idx] = self.generate_answers(sample,batch)
        return samples
    
    def generate_answers(self, sample: Dict,batch):
        generated_qs = sample['generated_questions']
        global_entity_dict = sample['entity_info']
        # prepare batch
        all_answers = []
        for gen_qs in generated_qs:
            # border case: no question asked.
            if len(gen_qs) == 0:
                all_answers.append({})
                continue
            cur_answers = {}
            for cur_qs in gen_qs:
                qs, entity = cur_qs # qs is a str. entity is also a str. may contain multiple entity connected by periods.
                is_multi = False
                if '(multi)' in entity:
                    is_multi = True

                entity_list = entity.replace('(multi)','').replace('(single)','').split('.')
                entity_list = [e.strip() for e in entity_list if e.strip()]
                
                cur_answers = get_all_answers(entity_list, qs, global_entity_dict, sample['img_path'], cur_answers,batch,is_multi)
            all_answers.append(cur_answers)

        sample['generated_answers'] = all_answers
        return sample
    
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
from typing import Dict
from models.utils import compute_iou
import torch
from sglang import function,user,assistant,system,gen
@function
def claimer(s,question,answer):
    s += system("You are an AI assistant that helps to summarize facts from given QA pairs. The summary should only contain facts related to the question, and you should ignore other facts unrelated to the question. The answer may mention a red bounding box, you should ignore the bounding box and do not mention it in the summary.")
    s += user('''Question: What is the man doing?
Answer: The man in the red bounding box appears to be in the process of cutting a cake. He is holding a knife and is leaning over a table with the cake on it. It looks like he is either about to make the first cut or has just finished cutting a piece. The setting suggests a casual, possibly festive occasion, given the presence of a cake and the man's attire, which includes a colorful shirt and a lei.''')
    s += assistant("Summary: The man is cutting a cake, holding a knife and leaning over a table with the cake on it.")
    s += user('''Question: Is this person sitting on the couch?
Answer: Yes, there is a person sitting on the couch in the image.''')
    s += assistant("Summary: There is a person sitting on the couch.")
    s += user('''Question:Is the woman holding the drink close to the camera?
Answer: No, the woman holding the drink in the red bounding box is far away from the camera.''')
    s += assistant("Summary: The woman holding the drink is far away from the camera.")
    s += user(f"Question: {question}\nAnswer: {answer}")
    s += assistant("Summary: "+gen('claim'))

def get_claim_or_prepare(question, answer,batch):
    if isinstance(question, str):
        question = [question]
    if isinstance(answer, str):
        answer = [answer]
    assert len(question) == len(answer)
    key = str((question,answer))
    if key in batch:
        return batch[key]['output']
    else:
        batch[key] = {'input':(question,answer),'output':[]}
        return []

def process_batch(batch,endpoint):
    flatten_batch = []
    for k in batch.keys():
        q_list = batch[k]['input'][0]
        a_list = batch[k]['input'][1]
        for q,a in zip(q_list,a_list):
            flatten_batch.append(dict(question=q,answer=a))
    states = claimer.run_batch(flatten_batch,temperature=0,max_new_tokens=256,backend=endpoint)
    state_iter = iter(states)
    for k in batch.keys():
        output = []
        q_list = batch[k]['input'][0]
        for _ in q_list:
            output.append(next(state_iter)['claim'])
        batch[k]['output'] = output

class ClaimGenerator:
    '''
        Input:
            dict: 
                'generated_questions': a list of 2-ele list, each [qs(str), involved entities(str)]
                                       each question is a list of 2 elements, [qs, entity]
                'generated_answers': An 1-d list of dict. Each dict in the list contains all the (qs, ans) tuple for each object instance.
                                {
                                    overall: [(qs, answer), ...]
                                    entity:  [
                                                [(qs, answer), ...]   (for instance 1 of this type of entity)
                                                    ...
                                             ]
                                }
        Output:
            dict:
                'claim': a global dict. merge 'generated_questions' and 'generated_answers' into sentence-level claims.
                            {
                                'specific':
                                    {
                                        entity 1: 2-d list. for each instance is a list: [claim1, claim2, ...]
                                        entity 2: 2-d list. for each instance is a list: [claim1, claim2, ...]   
                                    }
                                ...
                                'overall': 1-d list. 
                                'counting': 
                            }
    '''
    
    def __init__(self, endpoint):
        self.endpoint = endpoint
    def generate_batch_claim(self, samples: Dict):
        batch = {}
        # warm up batch
        for idx,sample in enumerate(samples):
            samples[idx] = self.generate_claim(sample,batch)

        process_batch(batch,self.endpoint)
        # assign value
        for idx,sample in enumerate(samples):
            samples[idx] = self.generate_claim(sample,batch)
        return samples

    def generate_claim(self, sample: Dict,batch):
        # claim from two parts. counting info and Q&A
        all_claim = {}
        
        # first part, Q&A
        generated_answers = sample['generated_answers']
        for answer_dict in generated_answers:
            for entity, answer_list in answer_dict.items():
                if entity == 'overall':
                    all_claim.setdefault('overall', [])
                    questions = []
                    answers = []
                    for qa_tuple in answer_list:
                        qs, ans = qa_tuple
                        questions.append(qs)
                        answers.append(ans)
                    clm = get_claim_or_prepare(questions, answers,batch)
                    all_claim['overall'] += clm
                else:
                    all_claim.setdefault('specific', {}).setdefault(entity, [])
                    for idx, entity_answer_list in enumerate(answer_list):
                        if idx + 1 > len(all_claim['specific'][entity]):
                            all_claim['specific'][entity].append([])
                        questions = []
                        answers = []
                        for qa_tuple in entity_answer_list:
                            qs, ans = qa_tuple
                            questions.append(qs)
                            answers.append(ans)
                        clm = get_claim_or_prepare(questions, answers,batch)
                        all_claim['specific'][entity][idx] += clm
                           
        # second part, counting info
        counting_claim = "Counting: \n"
        ent_bboxes = {}
        for entity, ent_info in sample['entity_info'].items():
            for idx, bbox in enumerate(ent_info['bbox']):
                ent_bboxes[f"{entity} {idx+1}"] = bbox
        
        IOU_THRESHOLD = 0.95
        ent_aliases = {}
        for s_ent,s_bbox in ent_bboxes.items():
            ent_aliases.setdefault(s_ent, [])
            for t_ent,t_bbox in ent_bboxes.items():
                if compute_iou(s_bbox, t_bbox) > IOU_THRESHOLD:
                    ent_aliases[s_ent].append(t_ent)


        for entity, ent_info in sample['entity_info'].items():
            counting_claim_list = []
            ent_counts = ent_info['total_count']
            if ent_counts == "unknown":
                continue
            if ent_counts == 0 and entity != "":
                counting_claim += f"There is no {entity}.\n"
                continue
            else:
                counting_claim += f"There {'are' if ent_counts>1 else 'is' } {ent_counts} {entity}.\n"
                box_claim_list = []
                for idx, bbox in enumerate(ent_info['bbox']):
                    ent_name = f"{entity} {idx+1}"
                    ent_alias = ', '.join(sorted(ent_aliases[ent_name]))
                    if len(ent_aliases[ent_name]) == 1:
                        final_name = ent_alias
                    else:
                        final_name = f"({ent_alias})"
                    box_claim_list.append(f"{final_name}")
                counting_claim += '\n'.join(box_claim_list) + '\n\n'
                
        all_claim['counting'] = counting_claim
        sample['claim'] = all_claim
        sample['ent_aliases'] = ent_aliases
        return sample
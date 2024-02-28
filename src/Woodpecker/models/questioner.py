from typing import Dict,List
from tqdm import tqdm
import openai
import time
import spacy
from sglang import function,system,user,assistant,gen
# Do not ask questions related to position or position relationship.

@function
def questioner(s,sentence,entity):
    s += system('''Given a sentence and some entities connnected by periods, you are required to ask some relevant questions about the specified entities involved in the sentence, so that the questions can help to verify the factuality of the sentence.
Questions may involve basic attributes such as colors, actions mentioned in the sentence. Do not ask questions involving object counts or the existence of object. For example, do not ask questions like "How many dogs are there in the image?" or "Is there a dog in the image?". Do not ask special interrogative sentences, which start with "How", "Why", "What", "Where", "When". You should Ask questions that can be easily answered by yes or no.
When asking questions about attributes, try to ask simple questions that only involve one entity. 
Ask questions that can be easily decided visually. Do not ask questions that require complex reasoning. The question should be easily answered by yes or no.
Do not ask semantically similar questions. Do not ask questions only about scenes or places.
Do not ask questions about uncertain or conjecture parts of the sentence, for example, the parts described with "maybe" or "likely", etc.
It is no need to cover all the specified entities. If there is no question to ask, simply output a 'None'.
When asking questions, do not assume the claims in the description as true in advance. Only ask questions relevant to the information in the sentence.
Only ask questions about common, specific and concrete entities. The entities involved in the questions are limited to the range within the given entities.
Output only one question in each line. For each line, first output the question, then a single '&', and finally entities involved in the question, still connected by periods if multiple entities are involved. If there is only one entity involved, but the entity indicates multiple instances, then add '(multi)' at the end, else if the entity indicates only one instance, then add '(single)' at the end. If there are multiple entities involved, just add '(multi)' after the last entity. For example, "Is this cat sleeping?&cat(single)", in which case the question is about a single cat, so we add '(single)', "Are the dogs barking?&dog(multi)", in which case the question is about multiple dogs, so we add '(multi)', "Is the man shaking hands with the woman?&man.woman(multi)", in which case the question involves multiple entities, so we add '(multi)' at the end. Note that you only need to add '(multi)' or '(single)' after the last entity, and do not need to add '(multi)' or '(single)' after the other entities.
If the question only involves one entity, you should use the word 'this' to refer the entity, like "this person", "this dog".
Again, Do not ask "How many" or "Is there" questions.''')
    s += user('''Sentence:
There are one black dog and two white cats in the image.

Entities:
dog.cat''')
    s += assistant('''Questions:
Is this a white cat?&cat(single)
Is this a black dog?&dog(single)''')
    s += user('''Sentence:
The man is wearing a baseball cap and appears to be smoking.

Entities:
man''')
    s += assistant('''Questions:
Is this man wearing a baseball cap?&man(single)
Is this man smoking?&man(single)''')
    s += user('''Sentence:
The image depicts a busy kitchen, with a man in a white apron. The man is standing in the middle of the kitchen.

Entities:
kitchen.man''')
    s += assistant('''Questions:
Is this man wearing a white apron?&man(single)
Is this man standing in the middle of the kitchen?&man.kitchen(multi)''')
    s += user('''Sentence:
There is a person partially visible in the background.

Entities:
person''')
    s += assistant('''Questions:
Is this person partially visible in the background?&person(single)''')
    s += user('''Sentence:
The woman and the men next to her are laughing.

Entities:
woman.man''')
    s += assistant('''Questions:
Is this woman laughing?&woman(single)
Are the men laughing?&man(multi)
Are the men standing next to the woman?&woman.man(multi)''')
    s += user('''Sentence:
There are several other people in the background of the photo, some of whom are more focused on the man and woman, while others appear to be engaged in party activities.

Entities:
person.man.woman''')
    s += assistant('''Questions:
Are there people in the background of the photo?&person(multi)
Are some of the people in the background focused on the man and woman?&person.man.woman(multi)''')
    s += user(f"Sentence:\n{sentence}\n\nEntities:\n{entity}")
    s += assistant("Questions:\n"+gen("question"))


def remove_duplicates(res):
    qs_set = set()
    output = []
    for s in res:
        qs, ent = s
        if qs in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(qs)
    return output

class Questioner:
    '''
        Input:
            For each splitted sentences:
                A sentence and list of existent objects. (only questions about existent objects)
        Output:
            For each splitted sentences:
                A list of 2-ele list: [[question, involved object type], [qs, obj], ...]         
    '''
    def __init__(self,endpoint):
    
        self.nlp = spacy.load("en_core_web_sm")
        self.endpoint = endpoint
    def generate_batch_questions(self, samples: List[Dict]):
        qs_list = []
        batch_entity = []
        batch_sent = []
        for sample in samples:
            sentences = sample['split_sents']
            global_entity_dict = sample['entity_info']
            global_entity_list = sample['entity_list']

            for ent_list, sent in zip(global_entity_list, sentences):
                exist_entity = [ent for ent in ent_list if ent in global_entity_dict and global_entity_dict[ent]['total_count'] != 0]

                # border case: no detection result for any entity. no question asked.
                if len(exist_entity)==0 :
                    qs_list.append([])
                    continue
                batch_entity.append('.'.join(exist_entity))
                batch_sent.append(sent)
                qs_list.append("placeholder")
                
        questions = self.get_batch_res(batch_entity, batch_sent)
        questions_iter=iter(questions)
        for idx,v in enumerate(qs_list):
            if v == "placeholder":
                qs_list[idx] = next(questions_iter)
        idx = 0
        for sample in samples:
            sample['generated_questions'] = []
            for sent in sample['split_sents']:
                sample['generated_questions'].append(qs_list[idx])
                idx += 1
        return samples
    
    def get_batch_res(self, batch_entity: List[str], batch_sent: List[str]):

        states = questioner.run_batch([{"sentence":sent,"entity":entity} for sent,entity in zip(batch_entity,batch_sent)],temperature=0,max_new_tokens=512,backend=self.endpoint)
        batch_res = []
        for state,sent,entity in zip(states,batch_sent,batch_entity):
            res = state['question'].splitlines()
            res = [s.split('&') for s in res if s.lower() != 'none']
            entity_list = entity.split('.')

            res = [s for s in res if len(s)==2]
            res = remove_duplicates(res)
            res = [s for s in res if set(s[1].replace('(single)','').replace('(multi)','').split('.')).issubset(set(entity_list)) ]
            batch_res.append(res)

        return batch_res
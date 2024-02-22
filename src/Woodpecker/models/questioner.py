from typing import Dict
from tqdm import tqdm
import openai
import time
import spacy

# Do not ask questions related to position or position relationship.
NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE='''Given a sentence and some entities connnected by periods, you are required to ask some relevant questions about the specified entities involved in the sentence, so that the questions can help to verify the factuality of the sentence.
Questions may involve basic attributes such as colors, actions mentioned in the sentence. Do not ask questions involving object counts or the existence of object. For example, do not ask questions like "How many dogs are there in the image?" or "Is there a dog in the image?".
When asking questions about attributes, try to ask simple questions that only involve one entity. 
Ask questions that can be easily decided visually. Do not ask questions that require complex reasoning.
Do not ask semantically similar questions. Do not ask questions only about scenes or places.
Do not ask questions about uncertain or conjecture parts of the sentence, for example, the parts described with "maybe" or "likely", etc.
It is no need to cover all the specified entities. If there is no question to ask, simply output a 'None'.
When asking questions, do not assume the claims in the description as true in advance. Only ask questions relevant to the information in the sentence.
Only ask questions about common, specific and concrete entities. The entities involved in the questions are limited to the range within the given entities.
Output only one question in each line. For each line, first output the question, then a single '&', and finally entities involved in the question, still connected by periods if multiple entities are involved. 
If the question only involves one entity, you should use the word 'this' to refer the entity, like "this person", "this dog".
Again, Do not ask "How many" or "Is there" questions.
Examples:
Sentence:
There are one black dog and two white cats in the image.

Entities:
dog.cat

Questions:
What color is this cat?&cat
What color is this dog?&dog

Sentence:
The man is wearing a baseball cap and appears to be smoking.

Entities:
man

Questions:
What is this man wearing?&man
What is this man doing?&man

Sentence:
The image depicts a busy kitchen, with a man in a white apron. The man is standing in the middle of the kitchen.

Entities:
kitchen.man

Questions:
What does this man wear?&man
Is the man standing in the middle of the kitchen?&man.kitchen

Sentence:
There is a person partially visible in the background.

Entities:
person

Questions:
Is this person partially visible in the background?&person

Sentence:
The woman and the man next to her are both laughing.

Entities:
woman.man

Questions:
Is this woman laughing?&woman
Is this man laughing?&man
Is the man stand next to the woman?&woman.man

Sentence:
There are several other people in the background of the photo, some of whom are more focused on the man and woman, while others appear to be engaged in party activities.

Entities:
person.man.woman

Questions:
Is this person in the background of the photo?&person
Are some of the people in the background focused on the man and woman?&person.man.woman

Sentence:
{sent}

Entities:
{entity}

Questions:'''


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
    def __init__(self, chatbot):

        self.chatbot =chatbot
    
        self.nlp = spacy.load("en_core_web_sm")
        
    def generate_questions(self, sample: Dict):
        sentences = sample['split_sents']
        global_entity_dict = sample['entity_info']
        global_entity_list = sample['entity_list']
        
        qs_list = []
        for ent_list, sent in zip(global_entity_list, sentences):
            exist_entity = [ent for ent in ent_list if ent in global_entity_dict and global_entity_dict[ent]['total_count'] > 0]
            
            # border case: no detection result for any entity. no question asked.
            if len(exist_entity)==0 :
                qs_list.append([])
                continue
            
            questions = self.get_res(self.nlp, '.'.join(exist_entity), sent)
            qs_list.append(questions)
        sample['generated_questions'] = qs_list
        return sample
    
    def get_res(self,nlp, entity: str, sent: str, max_tokens: int=1024,model='gpt-3.5-turbo'):
        content = PROMPT_TEMPLATE.format(sent=sent, entity=entity)

        response = self.chatbot.complete(content)

        res = response.splitlines()
        res = [s.split('&') for s in res if s.lower() != 'none']
        entity_list = entity.split('.')

        res = [s for s in res if len(s)==2]
        res = remove_duplicates(res)
        res = [s for s in res if set(s[1].split('.')).issubset(set(entity_list)) ]

        return res
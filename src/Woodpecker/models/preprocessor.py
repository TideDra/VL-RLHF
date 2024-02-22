from typing import  Dict
import spacy
import openai
import time
from tqdm import tqdm
import os
NUM_SECONDS_TO_SLEEP = 0.5

PROMPT_TEMPLATE='''Given a passage, you are required to replace pronouns such as "they" with the actual entities they refer to based on the context, then output the passage after replacement.
Only replace the pronouns if there is any, and do not change anything else in the original passage. 
If there is nothing to replace, then keep the original sentences unchanged, do not add any new sentence.
The modification should be as small as possible, and the output passage should have the same number of sentences as the original passage.

Examples:
Passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. He appears to be smoking.

Rewritten passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. The man appears to be smoking.

Passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right. 

Rewritten passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right. 


Passage:
{text}

Rewritten passage:'''


class PreProcessor:
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except:
            os.system("python -m spacy download en_core_web_lg")
            os.system("python -m spacy download en_core_web_md")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_lg')

    def get_split_sents(self, passage):
        doc = self.nlp(passage)
        split_sents = list(doc.sents)
        split_sents = [sent.text.strip() for sent in split_sents]
        return split_sents
    
    def generate_sentences(self, sample: Dict):
        rewritten_passage = self.get_output(sample['input_desc'])
        rew_split_sents = self.get_split_sents(rewritten_passage)
        
        orig_split_sents = self.get_split_sents(sample['input_desc'])
        
        sample['split_sents'] = rew_split_sents
        sample['orig_split_sents'] = orig_split_sents
        return sample

    def get_output(self,text: str, max_tokens: int=1024,model='gpt-3.5-turbo'):
        content = PROMPT_TEMPLATE.format(text=text)

        system = 'You are a language assistant that helps to rewrite a passage according to instructions.'
        response = self.chatbot.complete(content,system_message = system)

        return response
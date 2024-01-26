from typing import  Dict
from tqdm import tqdm
import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE='''Given a sentence, extract the entities within the sentence for me. 
Extract the common objects and summarize them as general categories without repetition, merge essentially similar objects.
Avoid extracting abstract or non-specific entities, such as lines, paintings, and so on. Make sure the entities can be detected by an object detection model.
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None". Notice again that the output should be in the singular form.

Examples:
Sentence:
The image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.

Output:
man.motorcycle

Sentence:
There are a few people around, including one person standing close to the motorcyclist and another person further away.

Output:
person.motorcyclist

Sentence:
No, there is no car in the image.

Output:
car

Sentence:
The image depicts a group of animals, with a black dog, a white kitten, and gray cats, sitting on a bed.

Output:
dog.cat.bed

Sentence:
{sentence}

Output:'''


class EntityExtractor:
    def __init__(self, chatbot):

        self.chatbot = chatbot
        
    
    def extract_entity(self, sample: Dict):
        extracted_entities = []
        for sent in sample['split_sents']:
            entity_str = self.get_res(sent)
            extracted_entities.append(entity_str)
        sample['named_entity'] = extracted_entities
        
        return sample

    def get_res(self,sent: str, max_tokens: int=1024,model='gpt-3.5-turbo'):
        content = PROMPT_TEMPLATE.format(sentence=sent)

        system =  'You are a language assistant that helps to rewrite a passage according to instructions.'

        response = self.chatbot.complete(content,system_message = system)

        return response
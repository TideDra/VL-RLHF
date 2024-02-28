from typing import  Dict, List
from sglang import function,user,assistant,system,gen
import inflect
@function
def extractor(s,sentence):
    s += system('''You are an AI assistant that extracts the entities within the given sentence. 
Extract the common objects and summarize them as general categories without repetition, merge essentially similar objects.
Avoid extracting abstract or non-specific entities, such as lines, paintings, and so on. Make sure the entities can be detected by an object detection model.
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None". Notice again that the output should be in the singular form.''')
    s += user("Sentence:\nThe image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.")
    s += assistant("Output:\nman.motorcycle")
    s += user("Sentence:\nThere are a few people around, including one person standing close to the motorcyclist and another person further away.")
    s += assistant("Output:\nperson.motorcyclist")
    s += user("Sentence:\nNo, there is no car in the image.")
    s += assistant("Output:\ncar")
    s += user("Sentence:\nThe image depicts a group of animals, with a black dog, a white kitten, and gray cats, sitting on a bed.")
    s += assistant("Output:\ndog.cat.bed")
    s += user(f"Sentence:\n{sentence}")
    s += assistant("Output:\n"+gen('entities'))

class EntityExtractor:
    def __init__(self,endpoint):
        self.singular_noun = inflect.engine().singular_noun
        self.endpoint = endpoint
    def extract_batch_entity(self, samples: List[Dict]):
        batch_sents = []
        for sample in samples:
            for sent in sample['split_sents']:
                batch_sents.append(sent)
        batch_entities = self.get_batch_res(batch_sents)
        idx = 0
        for sample in samples:
            extracted_entities = []
            for sent in sample['split_sents']:
                extracted_entities.append(batch_entities[idx])
                idx += 1
            sample['named_entity'] = extracted_entities
            
        return samples

    def get_batch_res(self,sent: List[str]):
        states = extractor.run_batch([{'sentence':s} for s in sent],temperature=0,max_new_tokens=1024,backend=self.endpoint)
        batch_entities = []
        for state in states:
            entities = [self.singular_noun(ent) if ent!='' and self.singular_noun(ent) else ent for ent in state['entities'].strip().split('.')]
            batch_entities.append('.'.join(entities).strip('.'))
        return batch_entities
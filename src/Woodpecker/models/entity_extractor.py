from typing import  Dict
from sglang import function,user,assistant,system,gen

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
    
    def extract_entity(self, sample: Dict):
        extracted_entities = []
        for sent in sample['split_sents']:
            entity_str = self.get_res(sent)
            extracted_entities.append(entity_str.strip('.'))
        sample['named_entity'] = extracted_entities
        return sample

    def get_res(self,sent: str,):
        state = extractor.run({'sentence':sent},temperature=0,max_new_tokens=1024)
        return state['entities'].strip()
from typing import  Dict
import spacy
import openai
import time
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.3
PROMPT_TEMPLATE='''Given a query, a passage and some supplementary information, you are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. The supplementary information may include some of the following parts:
    "Counting" information that specifies how many instances of a certain kind of entity exist, and their associated bounding boxes;
    "Specific" information that describes attribute information specific to each entity instance, including bounding boxes, colors, etc. The information is arranged in the form of "entity 1: [bbox] "info of this entity". Note that the entity in "Specific" information corresponds to that in the "Counting" information.
    "Overall" information that may involve information about multiple entity objects. 
2. Mark the words in the passage with <f> and </f> tag if they are inconsistent with the supplementary information, and then append the corrected words marked with <t> and </t> tag after the wrong words.
3. The structure of the corrected sentence should be consistent with the original sentence as possible. Try to correct as few words as possible, but make sure the sentence is fluent, natural and complete after correction, also considering the punctuation.
4. The number of entitie instances should match the number in the 'Counting' information. Also correct the number counts if the number stated in the original sentence does not match the counting information.
5. If the passage mentions entity or attribute that does not exist in the supplementary information, then mark the words and remove it, where 'remove' means the corrected words should be empty, or a simple '<t></t>' tag in other words. If the removed words contain some entities that truly exist, you should rewrite the sentence and contain the entities, instead of leaving an empty sentence. Note that the corrected sentence should be fluent, natural and complete after removal. So you should choose the words to remove carefully, also considering the punctuation.
6. In the refined passage, when describing entities mentioned in the "Specific" supplementary information, add their associated bounding boxes in parentheses right after them, in the form of "entity([bbox])". If multiple entities of the same kind are mentioned, then seperate the box with ';', in the form of "entity([bbox1];[bbox2])"
7. When deriving position relationships between entity instances, try to also use the bounding boxes information, which are represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x1, top left y1, bottom right x2, and bottom right y2. 
8. When giving refined passage, also pay attention to the given query. The refined passage should be reasonable answers to the query.
9. Note that instances of a certain category can also belong to its super-catergories. For example, a bus is also a car.

Examples:
Supplementary information:
Counting: 
There are 1 snowboard.
snowboard 1: [0.498, 0.555, 0.513, 0.577]

There are 1 person.
person 1: [0.496, 0.467, 0.535, 0.568]

There are 1 slope.
slope 1: [0.002, 0.002, 0.998, 0.998]

Specific:
person 1: [0.496, 0.467, 0.535, 0.568] The person is doing snowboarding.

slope 1: [0.002, 0.002, 0.998, 0.998] The slope is covered with snow.


Query:
Is there a snowboard in the image?

Passage:
No, there is no snowboard in the image. The image shows a person skiing down a snow-covered slope.

Refined passage: 
<f>No, there is no snowboard in the image.</f><t>Yes, there is a snowboard([0.498, 0.555, 0.513, 0.577]) in the image.</t> The image shows a person([0.496, 0.467, 0.535, 0.568]) <f>skiing down</f><t>doing snowboarding on</t> a snow-covered slope([0.002, 0.002, 0.998, 0.998]).

-------------------

Supplementary information:
Counting: 
There are 2 car.
car 1: [0.289, 0.637, 0.309, 0.661]
car 2: [0.315, 0.633, 0.324, 0.662]

There are 1 bus.
bus 1: [0.318, 0.092, 0.85, 0.963]

Specific:
bus 1: [0.318, 0.092, 0.85, 0.963] The bus is red.


Query:
Is there a car in the image?

Passage:
No, there is no car in the image. The image features a red double-decker bus.

Refined passage: 
<f>No, there is no car in the image.</f><t>Yes, there are cars([0.289, 0.637, 0.309, 0.661];[0.315, 0.633, 0.324, 0.662];[0.318, 0.092, 0.85, 0.963]) in the image.</t> The image features a red double-decker bus([0.318, 0.092, 0.85, 0.963]).

-------------------

Supplementary information:
Counting: 
There is no sports ball.

There are 1 soccer ball.
soccer ball 1: [0.682, 0.32, 0.748, 0.418]

Specific:
soccer ball 1: [0.682, 0.32, 0.748, 0.418] The soccer ball is in the image.


Query:
Is there a sports ball in the image?

Passage:
Yes, there is a sports ball in the image, and it appears to be a soccer ball.

Refined passage: 
Yes, there is a sports ball([0.682, 0.32, 0.748, 0.418]) in the image, and it appears to be a soccer ball([0.682, 0.32, 0.748, 0.418]).

-------------------

Supplementary information:
Counting: 
There are 1 ball.
ball 1: [0.682, 0.32, 0.748, 0.418]

Specific:
soccer ball 1: [0.682, 0.32, 0.748, 0.418] The soccer ball is in the image.


Query:
Is there a ball in the image?

Passage:
No, there is not a ball in this image.

Refined passage: 
<f>No</f><t>Yes</t>, there is <f>not</f><t></t>a ball([0.682, 0.32, 0.748, 0.418]) in this image.

-------------------

Supplementary information:
Counting: 
There are 4 dogs.
dogs 1: [0.668, 0.454, 0.728, 0.651]
dogs 2: [0.337, 0.529, 0.427, 0.747]
dogs 3: [0.438, 0.254, 0.49, 0.439]
dogs 4: [0.353, 0.325, 0.418, 0.535]


Query:
Are there four dogs in the image?

Passage:
No, there are only 3 dogs in the image.

Refined passage:
<f>No</f><t>Yes</t>, there are <f>only 3</f><t>four</t> dogs([0.668, 0.454, 0.728, 0.651];[0.337, 0.529, 0.427, 0.747];[0.438, 0.254, 0.49, 0.439];[0.353, 0.325, 0.418, 0.535]) in the image.

-------------------

Supplementary information:
Counting: 
There are 1 bicycle.
bicycle 1: [0.467, 0.555, 0.717, 0.746]

There are 2 trash bin.
trash bin 1: [0.145, 0.498, 0.321, 0.728]
trash bin 2: [0.319, 0.497, 0.483, 0.729]

Overall:
The bicycle is not on the right side of the trash bin.


Query:
Is the bicycle on the right side of the trash bin?

Passage:
No, the bicycle is not on the right side of the trash bin.

Refined passage: 
<f>No</f><t>Yes</t>, the bicycle([0.467, 0.555, 0.717, 0.746]) is <f>not</f><t></t>on the right side of the trash bin([0.145, 0.498, 0.321, 0.728];[0.319, 0.497, 0.483, 0.729]).

-------------------

Supplementary information:
Counting: 
There are 3 car.
car 1: [0.218, 0.306, 0.49, 0.646]
car 2: [0.879, 0.401, 1.0, 0.635]
car 3: [0.036, 0.0, 0.181, 0.172]

Specific:
car 1: [0.218, 0.306, 0.49, 0.646] The car is black.
car 2: [0.879, 0.401, 1.0, 0.635] The car is black.
car 3: [0.036, 0.0, 0.181, 0.172] The car is white.


Query:
Is there a black car in the image?

Passage:
Yes, there is a black car in the image.

Refined passage: 
Yes, there is a black car([0.218, 0.306, 0.49, 0.646]) in the image.

-------------------

Supplementary information:
Counting: 
There are 5 children.
children 1: [0.001, 0.216, 0.258, 0.935]
children 2: [0.573, 0.251, 0.98, 0.813]
children 3: [0.569, 0.313, 0.701, 0.662]
children 4: [0.31, 0.259, 0.582, 0.775]
children 5: [0.224, 0.283, 0.401, 0.663]

There is no shirts.

There are 1 field.
field 1: [0.001, 0.444, 0.998, 0.997]

There are 2 frisbees.
frisbees 1: [0.392, 0.473, 0.569, 0.713]
frisbees 2: [0.72, 0.486, 0.941, 0.76]

There is no backpack.

There is no handbag.

Specific:
children 1: [0.001, 0.216, 0.258, 0.935] The shirts are red and white. The children are holding a pig. The children are sitting on the grass.
children 2: [0.573, 0.251, 0.98, 0.813] The shirts are blue. The children are holding a frisbee. The children are sitting on the grass.
children 3: [0.569, 0.313, 0.701, 0.662] The shirts are blue. The children are holding a teddy bear. The children are sitting on the grass.
children 4: [0.31, 0.259, 0.582, 0.775] The shirts are brown. The children are holding a frisbee. The children are sitting on the grass.
children 5: [0.224, 0.283, 0.401, 0.663] The shirts are red. The children are holding a ball. The children are sitting on the grass.

frisbees 1: [0.392, 0.473, 0.569, 0.713] The frisbees are white.
frisbees 2: [0.72, 0.486, 0.941, 0.76] The frisbees are white.

Overall:
The children are sitting on the grass.
The children are in the grass.

Query:
Describe the image in detail.

Passage:
The image shows a group of young children, all wearing black shirts, sitting in a grassy field. They appear to be having a good time as they each hold two white frisbees. 

There's a total of seven children, ranging in age from young toddlers to older children, scattered throughout the scene. Some of the children are standing while others are sitting, enjoying their time in the field. 

In the background, there are several other items, such as a couple of backpacks placed near the field, and a handbag placed further back in the scene.

Refined passage: 
The image shows a group of young children([0.001, 0.216, 0.258, 0.935];[0.573, 0.251, 0.98, 0.813];[0.569, 0.313, 0.701, 0.662];[0.31, 0.259, 0.582, 0.775];[0.224, 0.283, 0.401, 0.663]), <f>all wearing black shirts, </f><t></t>sitting in a grassy field([0.001, 0.444, 0.998, 0.997]). <f>They appear to be having a good time as they each hold two white frisbees.</f><t>Some of the children([0.573, 0.251, 0.98, 0.813];[0.31, 0.259, 0.582, 0.775]) are holding a white frisbee([0.392, 0.473, 0.569, 0.713];[0.72, 0.486, 0.941, 0.76]).</t>

There's a total of <f>seven</f><t>five</t> children, ranging in age from young toddlers to older children, scattered throughout the scene. <f>Some of the children are standing while others are sitting</f><t>All the children are sitting</t>, enjoying their time in the field. 

<f>In the background, there are several other items, such as a couple of backpacks placed near the field, and a handbag placed further back in the scene.</f><t></t>

-------------------

Supplementary information:
Counting: 
There are 1 horse.
horse 1: [0.387, 0.227, 0.687, 0.798]

There are 1 rider.
rider 1: [0.479, 0.124, 0.654, 0.606]

There are 1 pond.
pond 1: [0.003, 0.601, 0.997, 0.995]

There is no race.

There are 1 water.
water 1: [0.003, 0.578, 0.997, 0.994]

There are 1 helmet.
helmet 1: [0.504, 0.126, 0.552, 0.179]

Specific:
horse 1: [0.387, 0.227, 0.687, 0.798] The horse is jumping through water.

rider 1: [0.479, 0.124, 0.654, 0.606] A horse is the rider seated on.

helmet 1: [0.504, 0.126, 0.552, 0.179] A helmet and a bridle is essential for safety during such events.

Overall:
A horse is navigating the muddy pond.
A horse and rider is navigating the muddy pond during a horse race.
The horse is galloping in the water.
The rider is seated on the horse.


Query:
Describe this image.

Passage:
The image features a horse and rider navigating a muddy pond during a horse race. The horse is galloping through the water, with its rider firmly seated on its back. The rider is wearing a helmet, which is essential for safety during such events.

There are several other horses in the scene, some closer to the foreground and others further back. The horses are spread out across the pond, with some closer to the left side and others closer to the right side. The overall atmosphere of the scene is lively and exciting, as the horses and riders compete in the muddy pond.

Refined passage: 
The image features a horse([0.387, 0.227, 0.687, 0.798]) and rider([0.479, 0.124, 0.654, 0.606]) navigating a muddy pond([0.003, 0.601, 0.997, 0.995]) during a horse race. The horse is galloping <f>through</f><t>in</t> the water, with its rider firmly seated on its back. The rider is wearing a helmet([0.504, 0.126, 0.552, 0.179]), which is essential for safety during such events.

<f>There are several other horses in the scene, some closer to the foreground and others further back. The horses are spread out across the pond, with some closer to the left side and others closer to the right side.</f><t></t>The overall atmosphere of the scene is lively and exciting<f>, as the horses and riders compete in the muddy pond</f><t></t>.

-------------------

Supplementary information:
{sup_info}
Query:
{query}

Passage:
{text}

Refined passage: '''


class Refiner:
    '''
        Input:
                'split_sents': 1-d list. Sentences splitted from the passage.
                'claim': 2-d list. Achieve by merging 'generated_questions' and 'generated answers' into sentence-level claims.
        Output:
                'output' : Final output, a refined passage.
    '''
    
    def __init__(self, chatbot):

        self.chatbot = chatbot
    

    def generate_output(self, sample: Dict):
        all_claim = sample['claim']
        global_entity_dict = sample['entity_info']
        
        # three parts: counting, specific, overall
        sup_info = ""
        # add counting info.
        sup_info += all_claim['counting']
        
        # add specific info.
        if 'specific' in all_claim and len(all_claim['specific']) > 0:
            sup_info += "Specific:\n"
            specific_claim_list = []
            for entity, instance_claim in all_claim['specific'].items():
                cur_entity_claim_list = []
                for idx, instance_claim_list in enumerate(instance_claim):
                    cur_inst_bbox = global_entity_dict[entity]['bbox'][idx]
                    cur_entity_claim_list.append(f"{entity} {idx + 1}: {cur_inst_bbox} " + ' '.join(instance_claim_list))
                specific_claim_list.append('\n'.join(cur_entity_claim_list))
            sup_info += '\n\n'.join(specific_claim_list)
            sup_info += '\n\n'
            
        # add overall info.
        if 'overall' in all_claim and len(all_claim['overall']) > 0:
            sup_info += "Overall:\n"
            sup_info += '\n'.join(all_claim['overall'])
            sup_info += '\n\n'
            
        sample['output'] = self.get_output(sample['query'], sample['input_desc'], sup_info)
        return sample

    def get_output(self,query: str, text: str, sup_info: str, max_tokens: int=4096,model='gpt-3.5-turbo-16k'):
        content = PROMPT_TEMPLATE.format(query=query, sup_info=sup_info, text=text)

        system = 'You are a language assistant that helps to rewrite a passage according to instructions.'
        response = self.chatbot.complete(content,system_message = system)

        return response
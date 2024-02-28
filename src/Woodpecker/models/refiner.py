from typing import  Dict
import spacy
import openai
import time
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.3
SYS_MESSAGE='''Given a query, a passage and some supplementary information, you are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. The supplementary information may include some of the following parts:
    "Counting" information that specifies how many instances of a certain kind of entity exist, the name of each instance will be listed in the form of "entity 1\nentity 2\nentity 3\n...";
    "Specific" information that describes attribute information specific to each entity instance. The information is arranged in the form of "entity 1: info of this entity". Note that the entity in "Specific" information corresponds to that in the "Counting" information.
    "Overall" information that may involve information about multiple entity objects. 
2. Mark the words in the passage with <f> and </f> tag if they are inconsistent with the supplementary information, and then append the corrected words marked with <t> and </t> tag after the wrong words. The corrected words must be the same topic as the original wrong words, which means you cannot just replace the wrong words with an arbitrary fact that provides different information from the original words.
3. The structure of the corrected sentence should be consistent with the original sentence as possible. Try to correct as few words as possible, but make sure the sentence is fluent, natural and complete after correction, also considering the punctuation.
4. The number of entitie instances should match the number in the 'Counting' information. Also correct the number counts if the number stated in the original sentence does not match the counting information.
5. If the passage mentions entity or attribute that does not exist in the supplementary information, then mark the words and remove it, where 'remove' means the corrected words should be empty, or a simple '<t></t>' tag in other words. If the removed words contain some entities that truly exist, you should rewrite the sentence and contain the entities, instead of leaving an empty sentence. Note that the corrected sentence should be fluent, natural and complete after removal. So you should choose the words to remove carefully, also considering the punctuation.
6. When giving refined passage, also pay attention to the given query. The refined passage should be reasonable answers to the query.
7. Note that instances of a certain category can also belong to its super-catergories. For example, a bus is also a car. Different name of the same instance will be shown together. For example, we use (car 1, bus 2) to denote that car 1 and bus 2 are the same instance. You should also consider this information and refine the passage correctly.
8. The specific information and the overall information may be conflict with the counting information. For example, the counting information says there is 1 car, but the specific information implies that there are 2 cars. In this case, you should ignore the specific information and trust the counting information.
9. You should only correct the given passage, do not add any extra sentences. Even if you think the supplement information is wrong, you do not need to correct it in the final output. Just make sure every wrong sentence in the output refined passage is from the original passage.
'''

EXAMPLE1=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 1 snowboard.
snowboard 1

There are 1 person.
person 1

There are 1 slope.
slope 1

Specific:
person 1: The person is doing snowboarding.
slope 1: The slope is covered with snow.


Query:
Is there a snowboard in the image?

Passage:
No, there is no snowboard in the image. The image shows a person skiing down a snow-covered slope.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content': "<f>No, there is no snowboard in the image.</f><t>Yes, there is a snowboard in the image.</t> The image shows a person <f>skiing down</f><t>doing snowboarding on</t> a snow-covered slope."
    }
]

EXAMPLE2=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 3 car.
car 1
car 2
(car 3, bus 1)

There are 1 bus.
(car 3, bus 1)

Specific:
(car 3, bus 1): The bus is red.


Query:
Is there a car in the image?

Passage:
No, there is no car in the image. The image features a red double-decker bus.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"<f>No, there is no car in the image.</f><t>Yes, there are cars in the image.</t> The image features a red double-decker bus."
    }
]

EXAMPLE3=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There is no sports ball.

There are 1 soccer ball.
soccer ball 1

Specific:
soccer ball 1: The soccer ball is in the image.


Query:
Is there a sports ball in the image?

Passage:
Yes, there is a sports ball in the image, and it appears to be a soccer ball.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"Yes, there is a sports ball in the image, and it appears to be a soccer ball."
    }
]

EXAMPLE4=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 1 ball.
(ball 1, soccer ball 1)

Specific:
(ball 1, soccer ball 1): The soccer ball is in the image.


Query:
Is there a ball in the image?

Passage:
No, there is not a ball in this image.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"<f>No</f><t>Yes</t>, there is <f>not</f><t></t>a ball([0.682, 0.32, 0.748, 0.418]) in this image."
    }
]


EXAMPLE5=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 4 dogs.
dogs 1
dogs 2
dogs 3
dogs 4


Query:
Are there four dogs in the image?

Passage:
No, there are only 3 dogs in the image.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"<f>No</f><t>Yes</t>, there are <f>only 3</f><t>four</t> dogs in the image."
    
    }
]

EXAMPLE6=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 1 bicycle.
bicycle 1

There are 2 trash bin.
trash bin 1
trash bin 2

Overall:
The bicycle is not on the right side of the trash bin.


Query:
Is the bicycle on the right side of the trash bin?

Passage:
No, the bicycle is not on the right side of the trash bin.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"<f>No</f><t>Yes</t>, the bicycle is <f>not</f><t></t>on the right side of the trash bin."
    }
]

EXAMPLE7=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 3 car.
car 1
car 2
car 3

Specific:
car 1: The car is black.
car 2: The car is black.
car 3: The car is white.


Query:
Is there a black car in the image?

Passage:
Yes, there is a black car in the image.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':"Yes, there is a black car in the image."
    }
]

EXAMPLE8=[
    {   'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 5 children.
children 1
children 2
children 3
children 4
children 5

There is no shirts.

There are 1 field.
field 1

There are 2 frisbees.
frisbees 1
frisbees 2

There is no backpack.

There is no handbag.

Specific:
children 1: The shirts are red and white. The children are holding a pig. The children are sitting on the grass.
children 2: The shirts are blue. The children are holding a frisbee. The children are sitting on the grass.
children 3: The shirts are blue. The children are holding a teddy bear. The children are sitting on the grass.
children 4: The shirts are brown. The children are holding a frisbee. The children are sitting on the grass.
children 5: The shirts are red. The children are holding a ball. The children are sitting on the grass.

frisbees 1: The frisbees are white.
frisbees 2: The frisbees are white.

Overall:
The children are sitting on the grass.
The children are in the grass.

Query:
Describe the image in detail.

Passage:
The image shows a group of young children, all wearing black shirts, sitting in a grassy field. They appear to be having a good time as they each hold two white frisbees. 

There's a total of seven children, ranging in age from young toddlers to older children, scattered throughout the scene. Some of the children are standing while others are sitting, enjoying their time in the field. 

In the background, there are several other items, such as a couple of backpacks placed near the field, and a handbag placed further back in the scene.

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':'''The image shows a group of young children, <f>all wearing black shirts, </f><t></t>sitting in a grassy field. <f>They appear to be having a good time as they each hold two white frisbees.</f><t>Some of the children are holding a white frisbee.</t>

There's a total of <f>seven</f><t>five</t> children, ranging in age from young toddlers to older children, scattered throughout the scene. <f>Some of the children are standing while others are sitting</f><t>All the children are sitting</t>, enjoying their time in the field. 

<f>In the background, there are several other items, such as a couple of backpacks placed near the field, and a handbag placed further back in the scene.</f><t></t>'''
    }
]

EXAMPLE9=[
    {
        'role':'user',
        'content':'''Supplementary information:
Counting: 
There are 1 horse.
horse 1

There are 1 rider.
rider 1

There are 1 pond.
pond 1

There is no race.

There are 1 water.
water 1

There are 1 helmet.
helmet 1

Specific:
horse 1: The horse is jumping through water.
rider 1: A horse is the rider seated on.
helmet 1: A helmet and a bridle is essential for safety during such events.

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

Refined passage:'''
    },
    {
        'role':'assistant',
        'content':'''The image features a horse and rider navigating a muddy pond during a horse race. The horse is galloping <f>through</f><t>in</t> the water, with its rider firmly seated on its back. The rider is wearing a helmet, which is essential for safety during such events.

<f>There are several other horses in the scene, some closer to the foreground and others further back. The horses are spread out across the pond, with some closer to the left side and others closer to the right side.</f><t></t>The overall atmosphere of the scene is lively and exciting<f>, as the horses and riders compete in the muddy pond</f><t></t>.'''
    }
]


few_shot_examples = EXAMPLE1+EXAMPLE2+EXAMPLE3+EXAMPLE4+EXAMPLE5+EXAMPLE6+EXAMPLE7+EXAMPLE8+EXAMPLE9


PROMPT_TEMPLATE = '''Supplementary information:
{sup_info}
Query:
{query}

Passage:
{text}

Refined passage:'''


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
        self.few_shot_examples = few_shot_examples
        self.sys_message = SYS_MESSAGE
    def generate_output(self, sample: Dict):
        all_claim = sample['claim']
        global_entity_dict = sample['entity_info']
        ent_aliases = sample['ent_aliases']
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
                    ent_name = f"{entity} {idx + 1}"
                    ent_alias = ', '.join(sorted(ent_aliases[ent_name]))
                    if len(ent_aliases[ent_name]) == 1:
                        final_name = ent_alias
                    else:
                        final_name = f"({ent_alias})"
                    cur_entity_claim_list.append(f"{final_name}: " + ' '.join(instance_claim_list))
                specific_claim_list.append('\n'.join(cur_entity_claim_list))
            sup_info += '\n'.join(specific_claim_list)
            sup_info += '\n\n'
            
        # add overall info.
        if 'overall' in all_claim and len(all_claim['overall']) > 0:
            sup_info += "Overall:\n"
            sup_info += '\n'.join(all_claim['overall'])
            sup_info += '\n\n'
            
        sample['output'] = self.get_output(sample['query'], sample['input_desc'], sup_info)
        return sample

    def get_output(self,query: str, text: str, sup_info: str):
        content = self.few_shot_examples+[{'role':'user','content':PROMPT_TEMPLATE.format(query=query, sup_info=sup_info, text=text)}]

        response = self.chatbot.complete(content,system_message=self.sys_message)

        return response
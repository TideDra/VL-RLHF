from GPTFactory import smart_build_factory
import json
import re
import os
from argparse import ArgumentParser
system_message = "You are an AI assistant good at understanding passages. You will be given a passage which may contain some factual mistakes marked with a pair of <f> and </f> tags, followed by the correction marked with a pair of <t> and </t> tags. The correction can be empty, which is expressed as '<t></t>', meaning the mistake should be safely deleted and no new content should be added. The mistakes and corrections are annotated by human, and this passage is noted as 'Human Annotated Passage'.\n You will be also given the same passage where the mistakes and corrections are annotated by another AI assistant, and this passage is noted as 'AI Annotated Passage'.\n Your task is to summarize and list the mistakes and the corresponding corrections of the two passages together. Each item contains a mistake and its annotator, as well as the corresponding corrections and their annotators.\n The annotator can only be 'Human','AI' or 'Human and AI' if both annotators agree on the mistake or the correction. If for the same mistake, the two annotator gives different corrections, you should just list the corrections and clarify the corresponding annotator. \nAn empty correction expressed by '<t></t>' should be written as '<Delete>'.\nNote that the same mistake or correction can be annotated in different ways by the two annotators, you should rely on the semantic meaning of the content to determine whether they are the same, in which case the annotator should be 'Human and AI'. It is possible that there is no mistake annotated by both of the two annotators, in which case you should only output 'No mistake found.'."
few_shot_examples = [
    {
        "role":"user",
        "content":'''Human Annotated Passage:
The image features a group of people enjoying a day at the beach, with a focus on several surfers riding waves in the ocean. There are <f>at least five</f><t>four</t> people in the water, some standing on surfboards, and others in various stages of surfing or preparing to ride a wave.\n\nIn total, there are <f>nine</f><t>three</t> surfboards visible in the scene, scattered throughout the area where people are surfing. The waves appear to be small, suitable for beginners or those looking for a more relaxed surfing experience. The atmosphere seems fun and lively, with everyone is likely enjoying themselves in the sun and water.

AI Annotated Passage:
The image features a group of people enjoying a day at the beach, with a focus on several surfers riding waves in the ocean. <f>There are at least five people in the water</f><t>There are three people in the water</t>, some standing on surfboards, and others in various stages of surfing or preparing to ride a wave.\n\n<f>In total, there are nine surfboards visible in the scene, scattered throughout the area where people are surfing.</f><t></t> The waves appear to be small, suitable for beginners or those looking for a more relaxed surfing experience. The atmosphere seems fun and lively, with everyone enjoying themselves in the sun and water.'''
    },
    {
        "role":"assistant",
        "content":'''1.
Mistake: There are at least five people in the water.
Annotator: Human and AI
Correction 1: There are four people in the water.
Annotator: Human
Correction 2: There are three people in the water.
Annotator: AI

2.
Mistake: In total, there are nine surfboards visible in the scene, scattered throughout the area where people are surfing.
Annotator: Human and AI
Correction 1: In total, there are three surfboards visible in the scene, scattered throughout the area where people are surfing.
Annotator: Human
Correction 2: <Delete>'''
    },
    {
        "role":"user",
        "content":'''Human Annotated Passage:
The image features a group of four giraffes standing together in a zoo enclosure. They are located near a building with a sloping roof and seem to be gathered around a tree. A zebra is also present in the enclosure, standing near the giraffes and appearing to be <f>grazing</f><t>not grazing</t>.\n\nThe enclosure is fenced in with wire, and there is a fence post close to the viewer, which provides a sense of depth in the scene. The animals are standing on top of dry grass, and the overall atmosphere suggests a peaceful coexistence between the giraffes and the zebra in their shared habitat.

AI Annotated Passage:
The image features a group of <f>four</f><t>five</t> giraffes standing together in a zoo enclosure. They are located near a building with a sloping roof and seem to be gathered around a tree. A zebra is also present in the enclosure, standing near the giraffes and <f>appearing to be grazing</f><t>not grazing</t>.\n\nThe enclosure is fenced in with wire, and there is a fence post close to the viewer, which provides a sense of depth in the scene. The animals are standing on top of dry grass, and the overall atmosphere suggests a peaceful coexistence between the giraffes and the zebra in their shared habitat.'''
    },
    {
        "role":"assistant",
        "content":'''1.
Mistake: There are four giraffes standing together in a zoo enclosure.
Annotator: Human
Correction 1: There are five giraffes standing together in a zoo enclosure.
Annotator: AI

2.
Mistake: The zebra is appearing to be grazing.
Annotator: Human and AI
Correction 1: The zebra is not grazing.
Annotator: Human and AI'''
    },
    {
        "role":"user",
        "content":'''Human Annotated Passage:
The image is a black and white photo of the Big Ben clock tower and the Palace of Westminster in London. The tower is prominently featured in the <f>middle</f><t>right side</t> of the scene, standing tall and elegant next to the palace. \n\nThe photo captures the essence of London's historical architecture and iconic landmarks. The view appears to be a panoramic one, offering a wide perspective of the cityscape. The black and white color scheme adds a touch of classic and timeless charm to the picture.

AI Annotated Passage:
The image is a black and white photo of the Big Ben clock tower and the Palace of Westminster in London. The tower is prominently featured in the middle of the scene, standing tall and elegant next to the palace. \n\nThe photo captures the essence of London's historical architecture and iconic landmarks. <f>The view appears to be a panoramic one, offering a wide perspective of the cityscape.</f><t></t> The black and white color scheme adds a touch of classic and timeless charm to the picture.'''
    },
    {
        "role":"assistant",
        "content":'''1.
Mistake: The tower is prominently featured in the middle of the scene.
Annotator: Human
Correction 1: The tower is prominently featured in the right side of the scene.
Annotator: Human

2.
Mistake: The view appears to be a panoramic one, offering a wide perspective of the cityscape.
Annotator: AI
Correction 1: <Delete>
Annotator: AI'''
    },
    {
        "role":"user",
        "content":'''Human Annotated Passage:
The image depicts a lively scene at a birthday party, with a man standing over a small cake on a dining table, cutting into it. Another person is sitting nearby on a couch, enjoying the party. There are several other people present in the room, engaged in conversation and celebrating the occasion.\n\nOn the table, there are various items such as a knife, a bowl, and a <f>cell phone</f><t>notebook</t>. Additionally, there <f>are two bottles</f><t>is one bottle</t> visible in the scene, likely containing beverages for the guests. The room is furnished with multiple <f>chairs and</f><t></t> couches, providing ample seating for the attendees. The party atmosphere is joyful, and everyone seems to be having a great time.

AI Annotated Passage:
The image depicts a lively scene at a birthday party, with a man standing over a small cake on a dining table, cutting into it. Another person is sitting nearby on a couch, enjoying the party. There are <f>several other people present</f><t>three other people</t> in the room, engaged in conversation and celebrating the occasion.\n\nOn the table, there are various items such as a knife, a bowl, and a cell phone. Additionally, there is <f>two bottles</f><t>one bottle</t> visible in the scene, likely containing beverages for the guests. The room is furnished with <f>multiple chairs and</f> couches, providing ample seating for the attendees. The party atmosphere is joyful, and everyone seems to be having a great time.'''
    },
    {
        "role":"assistant",
        "content":'''1.
Mistake: There are several other people present in the room.
Annotator: AI
Correction 1: There are three other people in the room.
Annotator: AI

2.
Mistake: There are two bottles visible in the scene.
Annotator: Human and AI
Correction 1: There is one bottle visible in the scene.
Annotator: Human and AI

3.
Mistake: The room is furnished with multiple chairs and couches.
Annotator: Human and AI
Correction 1: The room is furnished with couches.
Annotator: Human and AI'''
    },
    {
        "role":"user",
        "content":'''Human Annotated Passage:
The image features a glass of red wine sitting next to an open box of pizza. The glass of wine is almost full, with a deep red liquid inside. The pizza in the box has a thick crust and is covered in various toppings, such as olives and other ingredients. \n\nThe scene suggests a cozy meal setting, perhaps enjoyed by a person who has just come home after a long day and decided to treat themselves to some delicious pizza and a glass of wine. The combination of the two comfort foods and beverages creates a relaxing and satisfying atmosphere.

AI Annotated Passage:
The image features a glass of red wine sitting next to an open box of pizza. The glass of wine is almost full, with a deep red liquid inside. The pizza in the box has a thick crust and is covered in various toppings, such as olives and other ingredients. \n\nThe scene suggests a cozy meal setting, perhaps enjoyed by a person who has just come home after a long day and decided to treat themselves to some delicious pizza and a glass of wine. The combination of the two comfort foods and beverages creates a relaxing and satisfying atmosphere.'''
    },
    {
        "role":"assistant",
        "content":'''No mistake found.'''
    }
]

parser = ArgumentParser()
parser.add_argument("--result-path", type=str, default='/mnt/gozhang/VL-RLHF/src/Woodpecker/results.json')
args = parser.parse_args()

with open(args.result_path, 'r') as f:
    predictions = json.load(f)

api_info=[
                    {
                        'api_key': "455bd4c7b7a8448d8d4f81bb2b90f469",
                        'end_point':"https://test-gpt4-api-canada-east.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2024-02-15-preview"
                    },
                    {
                        'api_key': "b1485beab36d4796841878836f6b3575",
                        'end_point':"https://test-gpt-4-turbo-australia-east.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2024-02-15-preview"
                    },
                    {
                        'api_key':"30f1cb81f72d47af90c33f058e50fd89",
                        'end_point':"https://test-gpt-api-sweden-central.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2024-02-15-preview"
                    },
                    {
                        'api_key':"7a9bc8c30afc4ddebee73f30f032dee8",
                        'end_point':"https://testdeploy3.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-15-preview"
                    }
                ]

factory = smart_build_factory(api_info=api_info,model="gpt-4",service='azure',worker_num=32,tpm=8e4,rpm=480,temperature=0)

tasks = []
for idx,item in enumerate(predictions):
    refined = item['refined']
    pred = item['prediction']
    prompt = f"Human Annotated Passage:\n{refined}\n\nAI Annotated Passage:\n{pred}"
    messages = few_shot_examples+[{"role":"user","content":prompt}]
    tasks.append(dict(prompt=messages,system_message=system_message,id=idx))

results = factory.run_task(tasks)
results.sort(key=lambda x:x.id)


human_found_mistakes = 0
ai_found_mistakes = 0
overlap_mistakes = 0
overlap_corrections = 0
eval_results = []
for item,res in zip(predictions,results):
    item['eval_result'] = res.response
    output = res.response
    mistake_extractor1 = re.compile(r'\nMistake: .*\nAnnotator: .*\nCorrection 1: .*\nAnnotator: .*')
    mistake_extractor2 = re.compile(r'\nMistake: .*\nAnnotator: .*\nCorrection 1: .*\nAnnotator: .*\nCorrection 2: .*\nAnnotator: .*')
    mistake_annotator_extractor = re.compile(r'Mistake: .*\nAnnotator: .*')
    mistakes = mistake_extractor1.findall(output) + mistake_extractor2.findall(output)
    eval_results.append(item)
    if len(mistakes) == 0:
        continue
    else:
        for mis in mistakes:
            mistake_annotator = mistake_annotator_extractor.findall(mis)[0]
            if "Human and AI" in mistake_annotator:
                human_found_mistakes += 1
                ai_found_mistakes += 1
                overlap_mistakes += 1
                if "Correction 2" not in mis:
                    overlap_corrections += 1
            elif "Human" in mistake_annotator:
                human_found_mistakes += 1
            else:
                ai_found_mistakes += 1
precision = overlap_mistakes/ai_found_mistakes
recall = overlap_mistakes/human_found_mistakes
f1 = 2*precision*recall/(precision+recall)
print(f"Hallucination detection precision:{precision}")
print(f"Hallucination detection recall:{recall}")
print(f"Hallucination detection F1:{f1}")
print(f"Hallucination correction precision:{overlap_corrections/overlap_mistakes}")
os.makedirs('output',exist_ok=True)
eval_result_name = "output/"+args.result_path.split('/')[-1].replace('.json','_eval.json')
with open('eval_results.json', 'w') as f:
    json.dump(eval_results,f)
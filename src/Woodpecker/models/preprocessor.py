from typing import  Dict, List
import spacy
import os
from sglang import function,user,assistant,system,gen

@function
def rewriter(s,text):
    s += system('''You are an AI assistant that helps to rewrite given passages.
Given a passage, you are required to replace the subject in each sentence which is pronoun such as "they" with the actual entities they refer to based on the context, so that they can be uniquely refered to without the previous context. Then output the passage after replacement.
Only replace the pronouns if there is any, and do not change anything else in the original passage. 
If there is nothing to replace, then keep the original sentences unchanged, do not add any new sentence.
The modification should be as small as possible, and the output passage should have the same number of sentences as the original passage.''')
    s += user("Passage:\nThe image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. He appears to be smoking.")
    s += assistant("Rewritten passage:\nThe image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. The man appears to be smoking.")
    s += user("Passage:\nThe image features a single water bottle left on a green wooden bench. The bench has a yellow armrest and is situated on a sidewalk. The surrounding area appears to be a park, as there are trees visible in the background.\n\nThe water bottle, which is almost empty, is placed near the middle of the bench. It is possible that someone left the bottle behind or forgot to take it with them after enjoying a break in the park. The scene conveys a sense of neglect or carelessness, as the empty water bottle sits unattended on the park bench.")
    s += assistant("Rewritten passage:\nThe image features a single water bottle left on a green wooden bench. The bench has a yellow armrest and is situated on a sidewalk. The surrounding area appears to be a park, as there are trees visible in the background.\n\nThe water bottle, which is almost empty, is placed near the middle of the bench. It is possible that someone left the bottle behind or forgot to take it with them after enjoying a break in the park. The scene conveys a sense of neglect or carelessness, as the empty water bottle sits unattended on the park bench.")
    s += user("Passage:\nThe image features a person wearing snow skis, performing a trick on a small ramp or wooden obstacle in the snow. The person is in the air, having launched off the ramp, and appears to be having fun. \n\nThere are several other people in the scene, some of whom are also on skis. However, they are located further away from the main action and are not actively involved in the trick being performed. The snow covers the ground, creating a winter sports environment for the skiers.")
    s += assistant("Rewritten passage:\nThe image features a person wearing snow skis, performing a trick on a small ramp or wooden obstacle in the snow. The person is in the air, having launched off the ramp, and appears to be having fun. \n\nThere are several other people in the scene, some of whom are also on skis. However, the people other than the person in the air are located further away from the main action and are not actively involved in the trick being performed. The snow covers the ground, creating a winter sports environment for the skiers.")
    s += user(f"Passage:\n{text}")
    s += assistant("Rewritten passage:\n"+gen('rewrite'))

class PreProcessor:
    
    def __init__(self,endpoint):
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except:
            os.system("python -m spacy download en_core_web_lg")
            os.system("python -m spacy download en_core_web_md")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_lg')
        self.endpoint = endpoint
    def get_split_sents(self, passage):
        doc = self.nlp(passage)
        split_sents = list(doc.sents)
        split_sents = [sent.text.strip() for sent in split_sents]
        return split_sents
    
    def generate_batch_sentences(self, batch_sample: List[Dict]):
        batch_input_desc = [s['input_desc'] for s in batch_sample]
        batch_reweitten_passage = self.get_batch_output(batch_input_desc)
        for rewritten_passage, sample in zip(batch_reweitten_passage, batch_sample):
            rew_split_sents = self.get_split_sents(rewritten_passage)

            orig_split_sents = self.get_split_sents(sample['input_desc'])

            sample['split_sents'] = rew_split_sents
            sample['orig_split_sents'] = orig_split_sents
        return batch_sample

    def get_batch_output(self,text: List[str]):
        states = rewriter.run_batch([{'text':t} for t in text],temperature=0,max_new_tokens=1024,backend=self.endpoint)

        return [s['rewrite'] for s in states]
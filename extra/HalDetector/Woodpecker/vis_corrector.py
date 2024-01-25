from models.preprocessor import PreProcessor
from models.entity_extractor import EntityExtractor
from models.detector import Detector
from models.questioner import Questioner
from models.answerer import Answerer
from models.claim_generator import ClaimGenerator
from models.refiner import Refiner
from tqdm import tqdm
from typing import List, Dict
import time
from GPTFactory import GPT
class Corrector:
    def __init__(self, api_key=None,end_point=None,detector_config=None,detector_model_path=None,cache_dir=None,val_model_path=None,qa2c_model_path=None) -> None:
        # init all the model
        self.chatbot = GPT(model='abc',service='azure',api_key=api_key,end_point=end_point)
        self.preprocessor = PreProcessor(self.chatbot)
        self.entity_extractor = EntityExtractor(self.chatbot)
        self.detector = Detector(detector_config,detector_model_path,cache_dir)
        self.questioner = Questioner(self.chatbot)
        self.answerer = Answerer(val_model_path)
        self.claim_generator = ClaimGenerator(qa2c_model_path)
        self.refiner = Refiner(self.chatbot)
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'input_img': Path to a local image 
        '''
        print("start generating sentences...")
        sample = self.preprocessor.generate_sentences(sample)
        print("start extracting entities...")
        sample = self.entity_extractor.extract_entity(sample)
        print("start detecting objects...")
        sample = self.detector.detect_objects(sample)
        print("start generating questions...")
        sample = self.questioner.generate_questions(sample)
        print("start generating answers...")
        sample = self.answerer.generate_answers(sample)
        print(sample['generated_answers'])
        print("start generating claims...")
        sample = self.claim_generator.generate_claim(sample)
        print(sample['claim'])
        print("start refining...")
        sample = self.refiner.generate_output(sample)
        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
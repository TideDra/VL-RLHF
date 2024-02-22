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
import os
from GPTFactory import GPT
import torch
import gc
from sglang import set_default_backend, RuntimeEndpoint
from sglang.utils import http_request
class Corrector:
    def __init__(self, api_key=None,end_point=None,refiner_key=None,refiner_end_point=None,api_service='azure',detector_config=None,detector_model_path=None,cache_dir=None,val_model_endpoint=None,device='cuda') -> None:
        # init all the model
        self.chatbot = GPT(model='gpt-3.5-turbo',service=api_service,api_key=api_key,end_point=end_point,temperature=0.7)
        if refiner_key is not None and refiner_end_point is not None:
            self.refiner_chatbot = GPT(model='gpt-4',service=api_service,api_key=refiner_key,end_point=refiner_end_point,temperature=0.7)
        else:
            self.refiner_chatbot = self.chatbot
        self.runtime = RuntimeEndpoint(val_model_endpoint)
        set_default_backend(self.runtime)
        http_request(self.runtime.base_url+"/flush_cache")
        self.preprocessor = PreProcessor()
        self.entity_extractor = EntityExtractor()
        self.detector = Detector(detector_config,detector_model_path,cache_dir,device=device)
        self.questioner = Questioner(self.chatbot)
        self.answerer = Answerer(device=device)
        self.claim_generator = ClaimGenerator(device=device)
        self.refiner = Refiner(self.refiner_chatbot)
        self.cache_dir = cache_dir
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
        print("start generating claims...")
        sample = self.claim_generator.generate_claim(sample)
        print("start refining...")
        sample = self.refiner.generate_output(sample)
        print('done')
        os.system(f"rm -rf {self.cache_dir}")
        torch.cuda.empty_cache()
        gc.collect()
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
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
    def __init__(self,refiner_key=None,refiner_end_point=None,api_service='azure',detector_config=None,detector_model_path=None,cache_dir=None,val_model_endpoint=None,device='cuda') -> None:
        # init all the model

        self.refiner_chatbot = GPT(model='gpt-4',service=api_service,api_key=refiner_key,end_point=refiner_end_point,temperature=0.7)
        self.runtime = RuntimeEndpoint(val_model_endpoint)
        set_default_backend(self.runtime)
        http_request(self.runtime.base_url+"/flush_cache")
        self.preprocessor = PreProcessor()
        self.entity_extractor = EntityExtractor()
        self.detector = Detector(detector_config,detector_model_path,cache_dir,device=device)
        self.questioner = Questioner()
        self.answerer = Answerer(device=device)
        self.claim_generator = ClaimGenerator(device=device)
        self.refiner = Refiner(self.refiner_chatbot)
        self.cache_dir = cache_dir
        print("Finish loading models.")

    
    def correct(self, samples: List[Dict]):
        assert isinstance(samples, list), f"Only support batch input with tpye List[Dict], but got {type(sample)}"
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'input_img': Path to a local image 
        '''
        print("start generating sentences...")
        samples = self.preprocessor.generate_batch_sentences(samples)
        print("start extracting entities...")
        samples = self.entity_extractor.extract_batch_entity(samples)
        print("start detecting objects...")
        samples = self.detector.detect_batch_objects(samples)
        print("start generating questions...")
        samples = self.questioner.generate_batch_questions(samples)
        print("start generating answers...")
        samples = self.answerer.generate_batch_answers(samples)
        print("start generating claims...")
        samples = self.claim_generator.generate_batch_claim(samples)
        print("start refining...")
        sample = self.refiner.generate_output(samples[0])
        print('done')
        os.system(f"rm -rf {self.cache_dir}")
        torch.cuda.empty_cache()
        gc.collect()
        return sample
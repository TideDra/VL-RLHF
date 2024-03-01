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
from GPTFactory import GPT,smart_build_factory
import torch
import gc
from sglang import set_default_backend, RuntimeEndpoint
from sglang.utils import http_request
class Corrector:
    def __init__(self,api_info=None,api_service='azure',detector_config=None,detector_model_path=None,cache_dir=None,val_model_endpoint=None,chat_model_endpoint=None,device='cuda') -> None:
        # init all the model

        self.refiner_factory = smart_build_factory(api_info,model="gpt-4",service=api_service,worker_num=32,tpm=8e4,rpm=480,temperature=0.01)
        self.val_runtime = RuntimeEndpoint(val_model_endpoint)
        if chat_model_endpoint is not None:
            self.chat_runtime = RuntimeEndpoint(chat_model_endpoint)
            http_request(self.chat_runtime.base_url+"/flush_cache")
        else:
            self.chat_runtime = self.val_runtime
        http_request(self.val_runtime.base_url+"/flush_cache")
        self.preprocessor = PreProcessor(self.chat_runtime)
        self.entity_extractor = EntityExtractor(self.chat_runtime)
        self.detector = Detector(detector_config,detector_model_path,cache_dir,self.val_runtime,device=device)
        self.questioner = Questioner(self.chat_runtime)
        self.answerer = Answerer(self.val_runtime)
        self.claim_generator = ClaimGenerator(self.chat_runtime)
        self.refiner = Refiner(self.refiner_factory)
        self.cache_dir = cache_dir
        print("Finish loading models.")

    
    def correct(self, samples: List[Dict]):
        assert isinstance(samples, list), f"Only support batch input with tpye List[Dict], but got {type(samples)}"
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'img_path': Path to a local image 
            'query': Query of the input_desc
        '''
        print('preprocessing...')
        samples = self.preprocessor.generate_batch_sentences(samples)
        print('extracting entities...')
        samples = self.entity_extractor.extract_batch_entity(samples)
        print('detecting objects...')
        samples = self.detector.detect_batch_objects(samples)
        print('generating questions...')
        samples = self.questioner.generate_batch_questions(samples)
        print('generating answers...')
        samples = self.answerer.generate_batch_answers(samples)
        print('generating claims...')
        samples = self.claim_generator.generate_batch_claim(samples)
        print('refining...')
        samples = self.refiner.generate_batch_output(samples)

        os.system(f"rm -rf {self.cache_dir}")
        torch.cuda.empty_cache()
        gc.collect()
        return samples
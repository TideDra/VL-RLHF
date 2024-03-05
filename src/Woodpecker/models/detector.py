import os

from typing import Dict, List
import numpy as np

import shortuuid
from itertools import cycle
import torch
from .utils import compute_iou
from mmdet.apis import DetInferencer
from mmcv import imread
from PIL import Image, ImageDraw
from .utils import image_qa
import gc
from threading import Thread,Lock
SCORE_THRESHOLD = 0.3    # used to filter out low-score object.
AREA_THRESHOLD = 0.001   # used to filter out too small object.
IOU_THRESHOLD = 0.95     # used to filter the same instance. greater than threshold means the same instance

def alreay_exist(detected_boxes, norm_box):
    if(len(detected_boxes)==0):
        return False

    if any([compute_iou(norm_box, box) > IOU_THRESHOLD for box in detected_boxes]):
        return True
    return False
    
def extract_detection(global_entity_dict, boxes, phrases, image_source, cache_dir, sample):
        
    h, w, _ = image_source.shape
    xyxy = np.array(boxes)
    normed_xyxy = np.around(np.clip(xyxy / np.array([w, h, w, h]), 0., 1.), 3).tolist()
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for entity, box, norm_box in zip(phrases, xyxy, normed_xyxy):
        # filter out too small object
        thre = sample['area_threshold'] if 'area_threshold' in sample else AREA_THRESHOLD
        if (norm_box[2]-norm_box[0]) * (norm_box[3]-norm_box[1]) < thre:
            continue
        # filter out object already in the dict
        if alreay_exist(global_entity_dict[entity]['bbox'], norm_box):
           continue 
       
        # add instance, including the cropped_pic & its original bbox
        crop_id = shortuuid.uuid()
        image = Image.fromarray(image_source)
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="red", width=4)
        crop_path = os.path.join(cache_dir, f"{crop_id}.png")
        image.save(crop_path)
        
        global_entity_dict[entity]['total_count'] += 1
        global_entity_dict[entity]['crop_path'].append(crop_path)
        global_entity_dict[entity]['bbox'].append(norm_box)    # [x1, y1, x2, y2] coordinate of left-top and right-bottom corner
        
    return global_entity_dict

def double_check(samples,endpoint,minibatch_size):
    maybe_entities = []
    for sample in samples:
        global_entity_dict = sample['entity_info']
        img_path = sample['img_path']
    
        for entity, info in global_entity_dict.items():
            if info['total_count'] == 0:
                maybe_entities.append({
                    'image_path': img_path,
                    'entity':entity
                })

    states = []
    for i in range(0, len(maybe_entities), minibatch_size):
        mini_batch = maybe_entities[i:min(i+minibatch_size,len(maybe_entities))]

        states.extend(image_qa.run_batch(
            [{"image_path":v['image_path'],"question":f"Is there any {v['entity']} in the image? Please answer yes or no."} for v in mini_batch],
            temperature=0,
            max_new_tokens=16,
            backend=endpoint
        ))
    states_iter = iter(states)
    for sample in samples:
        global_entity_dict = sample['entity_info']
        img_path = sample['img_path']
    
        for entity, info in global_entity_dict.items():
            if info['total_count'] == 0:
                state = next(states_iter)
                answer = state['answer']
                if "yes" in answer.lower():
                    global_entity_dict[entity]['total_count'] = 'unknown'

class Detector:
    '''
        Input: 
            img_path: str.
            named_entity: A list of str. Each in a format: obj1.obj2.obj3...
        Output:
            A list of dict, each dict corresponds to a series of objs.
            key: obj name. (obj1 | obj2 | obj3)
            value:
                {
                    total_count: detected counts of that obj.
                    
                    crop_path: a list of str, denoting the path to cached intermediate file, i.e., cropped out region of that obj.
                        Note: if total_count > 1, may use the whole image in the following steps.
                }
    '''
    def __init__(self, detector_config,detector_model_path, cache_dir,endpoint,minibatch_size=16,devices='0,1'):
        
        self.devices = devices.split(',')
        self.model_pool = []
        self.model_init_lock = Lock()
        threads = []
        for device in self.devices:
            t = Thread(target=self.init_model_worker,args=(detector_config,detector_model_path,device))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.cache_dir = cache_dir
        self.endpoint = endpoint
        self.minibatch_size = minibatch_size
    
    def init_model_worker(self,model,weights,device):
        inferencer = DetInferencer(model=model,weights=weights,device=f'cuda:{device}',palette='random',show_progress=False)
        with self.model_init_lock:
            self.model_pool.append(inferencer)

    def detect_objects(self, sample: Dict,device_idx:int):
        img_path = sample['img_path']
        image_source = imread(img_path)
        extracted_entities = sample['named_entity']
        
        global_entity_dict = {} # key=entity type name. value = {'total_count':int, 'crop_path':list, 'bbox':list of list(4-ele).}
        global_entity_list = [] # save all the entity type name for each sentence.
        for entity_str in extracted_entities:
            # border case: nothing to extract
            if 'none' in entity_str.lower():
                continue
            entity_list = entity_str.split('.')
            for ent in entity_list:
                global_entity_dict.setdefault(ent, {}).setdefault('total_count', 0)
                global_entity_dict.setdefault(ent, {}).setdefault('crop_path', [])
                global_entity_dict.setdefault(ent, {}).setdefault('bbox', [])
                
            global_entity_list.append(entity_list)
        
            predictions = self.model_pool[device_idx](img_path,texts=' . '.join(entity_list),custom_entities=True,pred_score_thr=SCORE_THRESHOLD)['predictions'][0]
            phrases = []
            boxes = []
            for label,score,bbox in zip(predictions['labels'],predictions['scores'],predictions['bboxes']):
                if score < SCORE_THRESHOLD:
                    # score is sorted, so we can break here.
                    break
                phrases.append(entity_list[label])
                boxes.append(bbox)
            torch.cuda.empty_cache()
            gc.collect() 
            if len(boxes) == 0:
                continue
            global_entity_dict = extract_detection(global_entity_dict, boxes, phrases, image_source, self.cache_dir, sample)
        
        #double_check(global_entity_dict, img_path,self.endpoint)
        sample['entity_info'] = global_entity_dict
        sample['entity_list'] = global_entity_list
        return sample

    def detect_objects_worker(self,samples:List[Dict],device_idx:int):
        for idx,sample in enumerate(samples):
            samples[idx] = self.detect_objects(sample,device_idx)

    def detect_batch_objects(self,samples:List[Dict]):
        device_num = len(self.devices)
        tasks = [[]]*device_num
        device_iter = cycle(range(device_num))
        for sample in samples:
            tasks[next(device_iter)].append(sample)
        threads = []
        for idx in range(device_num):
            t = Thread(target=self.detect_objects_worker,args=(tasks[idx],idx))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        double_check(samples,self.endpoint,self.minibatch_size)
        return samples
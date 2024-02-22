from  multiprocessing import Process,Queue,set_start_method

import sys
sys.path.append('/mnt/gozhang/code/VL-RLHF/')
import os

from vis_corrector import Corrector

args = {
    'api_key': "7a9bc8c30afc4ddebee73f30f032dee8",
    'end_point':"https://testdeploy3.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-07-01-preview",
    'val_model_path': "/mnt/gozhang/code/VL-RLHF/ckpts/Qwen-VL-Chat",
    'qa2c_model_path': "/mnt/gozhang/code/VL-RLHF/ckpts/zerofec-qa2claim-t5-base",
    'detector_config': "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    'detector_model_path': "../../GroundingDINO/weights/groundingdino_swint_ogc.pth",
    'cache_dir': "./cache_dir/",
    'api_service': "azure"
}


sample = {
'img_path': "../../../data_dir/coco2017/test2017/000000497687.jpg",
'input_desc': "The image features a group of three friends standing close together, possibly at a bar or a party. Two of the friends are women, and they are both holding glasses with drinks in their hands. One of the women is holding a drink in her hand up close to the camera, while the other has her drink slightly farther away. The third friend is a young man wearing a tie, standing between the two women.\n\nThe group appears to be enjoying themselves, with smiles on their faces, and they are facing the camera. In the background, there is another person partially visible. The scene conveys a sense of camaraderie and fun among the friends.",
'query': "Describe this picture in detail."
}


def worker(device,corrector_args,job_queue,result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    corrector = Corrector(**corrector_args)

    while True:
        sample = job_queue.get()
        result = corrector.correct(sample)
        result_queue.put(result)
if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except:
        pass
    job_q = Queue()
    result_q = Queue()
    
    pool_list = []
    for i in range(2):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = Process(target=worker,args=(str(i),args,job_q,result_q))
        pool_list.append(p)
        p.start()

    
    job_q.put(sample)
    job_q.put(sample)
    print(result_q.get()['output'])
    print(result_q.get()['output'])
    for i in range(4):
        pool_list[i].join()
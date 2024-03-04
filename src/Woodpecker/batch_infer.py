from vis_corrector import Corrector
from  multiprocessing import Process,Queue,set_start_method

import json
import tqdm
import os
from copy import deepcopy
from time import sleep
args = {
    'api_info':[
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
                ],
    'val_model_config': {
        "model_path":"/mnt/gozhang/ckpts/llava-v1.6-34b",
        "tokenizer_path":"/mnt/gozhang/ckpts/llava-v1.6-34b-tokenizer",
        "tp_size":2,
    },
    'chat_model_config': None,
    'detector_config': "./mmGD_config/grounding_dino_swin-t_pretrain_obj365.py",
    'detector_model_path': "/mnt/gozhang/ckpts/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth",
    'cache_dir': "./cache_dir/",
    'api_service': "azure",
    'minibatch_size':16
}

def worker(device,corrector_args,job_queue,result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    new_arg = deepcopy(corrector_args)
    new_arg['devices']=device
    new_arg['cache_dir'] = f"./cache_dir/{device}/"
    corrector = Corrector(**new_arg)
    result_queue.put("ready")
    while True:
        batch = job_queue.get()
        input_batch = [{'img_path':os.path.join("/mnt/gozhang/code/VL-RLHF/data_dir/",s['image']),'input_desc':s['caption'],'query':"Describe this image in detail."} for s in batch]
        try:
            result = corrector.correct(input_batch)
        except Exception as e:
            print(e)
            raise
        for sample,res in zip(batch,result):
            sample['prediction'] = res['output']
        result_queue.put(batch)

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except:
        pass
    job_q = Queue()
    result_q = Queue()
    
    pool_list = []
    gpu_num = 4
    gpu_per_worker = 2
    assert gpu_num % gpu_per_worker == 0
    for i in range(0,gpu_num,gpu_per_worker):
        devices = ",".join([str(j) for j in range(i,i+gpu_per_worker)])
        p = Process(target=worker,args=(devices,args,job_q,result_q))
        pool_list.append(p)
        p.start()
        if result_q.get() != "ready":
            raise Exception("Worker not ready.")
    hdbench_path = "/mnt/gozhang/code/VL-RLHF/src/hdbench/data/hdbench_cleaned.json"
    with open(hdbench_path, "r") as f:
        hdbench = json.load(f)
    batch_size=16
    result_file = "llava1.6-34b_hdbench.json"
    bar = tqdm.tqdm(total=len(hdbench))
    for i in range(0, len(hdbench), batch_size):
        batch = hdbench[i:min(i+batch_size,len(hdbench))]
        job_q.put(batch)
    
    results = []
    while len(results) < len(hdbench):
        result = result_q.get()
        results.extend(result)
        bar.update(len(result))

    with open(result_file, "w") as f:
        json.dump(results, f)
    bar.close()
    for p in pool_list:
        p.terminate()
        p.join()
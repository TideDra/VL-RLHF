from vis_corrector import Corrector
from models.refiner import SYS_MESSAGE as REF_SYS_MESSAGE
from models.refiner import few_shot_examples as ref_few_shot_examples
import json
import tqdm
import os
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
    'val_model_endpoint': "http://localhost:30000",
    'chat_model_endpoint': None,
    'detector_config': "./GroundingDINO_SwinT_OGC.py",
    'detector_model_path': "/mnt/gozhang/ckpts/groundingdino_swint_ogc.pth",
    'cache_dir': "./cache_dir/",
    'api_service': "azure",
}

if __name__ == "__main__":
    corrector = Corrector(**args)
    hdbench_path = "/mnt/gozhang/VL-RLHF/src/vlrlhf/eval/captioning/results/qwenvl_test2017.json"
    with open(hdbench_path, "r") as f:
        hdbench = json.load(f)
    batch_size=16
    result_file = "results.json"
    bar = tqdm.tqdm(total=len(hdbench))
    for i in range(0, len(hdbench), batch_size):
        batch = hdbench[i:min(i+batch_size,len(hdbench))]
        results = corrector.correct([{'img_path':os.path.join("/mnt/gozhang/VL-RLHF",s['image'][2:]),'input_desc':s['caption'],'query':"Describe this image in detail."} for s in batch])
        with open(result_file, "a") as f:
            for res in results:
                f.write(json.dumps({"image": res["img_path"],"input":res["input_desc"] ,"output": res["output"]}))
                f.write("\n")
        bar.update(len(results))
    bar.close()
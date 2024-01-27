from  multiprocessing import Process,Queue,set_start_method
import gradio as gr
import sys
sys.path.append('/mnt/gozhang/code/VL-RLHF/')
import os
from copy import deepcopy
from vis_corrector import Corrector
from models.refiner import SYS_MESSAGE as REF_SYS_MESSAGE
from models.refiner import few_shot_examples as ref_few_shot_examples

args = {
    'api_key':"7a9bc8c30afc4ddebee73f30f032dee8",
    'end_point':"https://testdeploy3.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-07-01-preview",
    'refiner_key': "05d739b8fe5141699aa0ab8b8cdacfa2",
    'refiner_end_point':"https://test-gpt-api-switzerlan-north.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-07-01-preview",
    'val_model_path': "/mnt/gozhang/code/VL-RLHF/ckpts/Qwen-VL-Chat",
    'qa2c_model_path': "/mnt/gozhang/code/VL-RLHF/ckpts/zerofec-qa2claim-t5-base",
    'detector_config': "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    'detector_model_path': "../../GroundingDINO/weights/groundingdino_swint_ogc.pth",
    'cache_dir': "./cache_dir/",
    'api_service': "azure"
}

def worker(device,corrector_args,job_queue,result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    new_arg = deepcopy(corrector_args)
    new_arg['cache_dir'] = f"./cache_dir/{device}/"
    corrector = Corrector(**new_arg)

    while True:
        sample = job_queue.get()
        refiner_sys_message = sample.pop('refiner_sys_message')
        refiner_examples = sample.pop('refiner_examples')
        corrector.refiner.few_shot_examples = refiner_examples
        corrector.refiner.sys_message = refiner_sys_message
        result = corrector.correct(sample)
        result_queue.put(result)

def inference(img,text,query,refiner_sys_message,*refiner_examples):
    new_few_shot_examples = deepcopy(ref_few_shot_examples)
    for idx, content in enumerate(refiner_examples):
        new_few_shot_examples[idx]['content'] = content
    sample = {
    'img_path': img,
    'input_desc': text,
    'query': query,
    'refiner_sys_message': refiner_sys_message,
    "refiner_examples": new_few_shot_examples,
    }
    job_q.put(sample)
    result = result_q.get()
    rest_text = result['output']
    split_text = []
    while len(rest_text)>0:
        f_start = rest_text.find('<f>')
        if f_start == -1:
            split_text.append((rest_text,None))
            break
        split_text.append((rest_text[:f_start],None))
        f_end = rest_text.find('</f>')
        split_text.append((rest_text[f_start:f_end+4],"False"))
        rest_text = rest_text[f_end+4:]
        t_start = rest_text.find('<t>')
        t_end = rest_text.find('</t>')
        split_text.append((rest_text[t_start:t_end+4],"True"))
        rest_text = rest_text[t_end+4:]
    result.pop('ent_aliases')
    result.pop('img_path')
    result.pop('input_desc')
    result.pop('query')
    return split_text,result

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except:
        pass
    job_q = Queue()
    result_q = Queue()
    
    pool_list = []
    gpu_num = 2
    for i in range(gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = Process(target=worker,args=(str(i),args,job_q,result_q))
        pool_list.append(p)
        p.start()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath",label="Image")
                query = gr.Textbox(lines=3,value="Describe this picture in detail.",label="Query")
                text = gr.Textbox(label="Answer",placeholder="Answer to be detected",lines=10)
                btn = gr.Button(value="Submit")
                with gr.Accordion(label="Refiner Setting"):
                    refiner_sys_message = gr.Textbox(lines=10,label="Refiner System Message",value=REF_SYS_MESSAGE)
                    refiner_examples_el = []
                    with gr.Group():
                        for idx,(user,assistant) in enumerate(zip(ref_few_shot_examples[::2],ref_few_shot_examples[1::2])):
                            prompt = gr.Textbox(lines=5,label=f"Prompt {idx}",value=user['content'])
                            response = gr.Textbox(lines=5,label=f"Response {idx}",value=assistant['content'])
                            refiner_examples_el += [prompt,response]
            with gr.Column():
                output = gr.HighlightedText(label="Output",show_legend=True,combine_adjacent=True,color_map={"True":"green","False":"red"})
                with gr.Accordion(label="Intermediate Detail"):
                    intermediate_detail = gr.JSON()
        btn.click(inference,inputs=[img,text,query,refiner_sys_message]+refiner_examples_el,outputs=[output,intermediate_detail])
    demo.launch(share=True,max_threads=4)

        

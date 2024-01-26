from  multiprocessing import Process,Queue,set_start_method
import gradio as gr
import sys
sys.path.append('/mnt/gozhang/code/VL-RLHF/')
import os
from copy import deepcopy
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

def worker(device,corrector_args,job_queue,result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    new_arg = deepcopy(corrector_args)
    new_arg['cache_dir'] = f"./cache_dir/{device}/"
    corrector = Corrector(**corrector_args)

    while True:
        sample = job_queue.get()
        result = corrector.correct(sample)
        result_queue.put(result)

def inference(img,text,query):
    sample = {
    'img_path': img,
    'input_desc': text,
    'query': query
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
    return split_text

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except:
        pass
    job_q = Queue()
    result_q = Queue()
    
    pool_list = []
    for i in range(4):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = Process(target=worker,args=(str(i),args,job_q,result_q))
        pool_list.append(p)
        p.start()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath",label="Image")
                query = gr.Textbox(lines=3,value="Describe this picture in detail.",label="Query")
                text = gr.Textbox(label="Answer")
                btn = gr.Button(value="Submit")
            with gr.Column():
                output = gr.HighlightedText(label="Output",show_legend=True,combine_adjacent=True,color_map={"True":"green","False":"red"})
        btn.click(inference,inputs=[img,text,query],outputs=output)
    demo.launch(share=True,max_threads=4)

        

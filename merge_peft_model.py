from utils.auto_load import MyAutoModel
from peft import PeftModel
from argparse import ArgumentParser
import os
from loguru import logger
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path',type=str,default='ckpts/Qwen-VL-Chat')
    parser.add_argument('--adapter_path',type=str)
    args = parser.parse_args()
    model = MyAutoModel.from_pretrained(args.model_path)
    peft_model = PeftModel.from_pretrained(model,args.adapter_path)
    logger.info('Merging and unloading model...')
    merged_model = peft_model.merge_and_unload()
    save_path = os.path.join(args.adapter_path,'merged')
    os.makedirs(save_path,exist_ok=True)
    logger.info(f'Saving merged model to {save_path}...')
    merged_model.save_pretrained(save_path)
    logger.success('Done!')

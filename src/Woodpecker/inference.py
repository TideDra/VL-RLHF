from vis_corrector import Corrector
from types import SimpleNamespace
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'Woodpecker: Hallucination Correction for MLLMs Hallucination Correction for MLLMs'.")
    parser.add_argument('--image-path', type=str, help="file path for the text to be corrected.")
    parser.add_argument('--query', type=str, help="text query for MLLM")
    parser.add_argument('--text', type=str, help="text from MLLM to be corrected")
    parser.add_argument('--cache-dir', type=str, help="dir for caching intermediate image",
                        default='./cache_dir')
    
    parser.add_argument('--detector-config', type=str, help="Path to the detector config, \
                        in the form of 'path/to/GroundingDINO_SwinT_OGC.py' ")
    parser.add_argument('--detector-model', type=str, help="Path to the detector checkpoint, \
                        in the form of 'path/to/groundingdino_swint_ogc.pth' ")

    parser.add_argument('--api-key', type=str, help="API key for GPT service.")
    parser.add_argument('--end-point', type=str, help="API base link for GPT service.")
    parser.add_argument('--api-service', type=str, help="GPT API service name. 'azure' or 'oai'.", default='azure')
    parser.add_argument('--val-model-path', type=str, help="Path to the validation model checkpoint")
    parser.add_argument('--qa2c-model-path', type=str, help="Path to the qa2claim model checkpoint")
    args = parser.parse_args()
    
    args_dict = {
        'api_key': args.api_key if args.api_key else "",
        'end_point':args.end_point if args.end_point else "",
        'val_model_path': args.val_model_path,
        'qa2c_model_path': args.qa2c_model_path,
        'detector_config':args.detector_config,
        'detector_model_path':args.detector_model,
        'cache_dir': args.cache_dir,
        'api_service': args.api_service

}

    corrector = Corrector(**args_dict)

    sample = {
    'img_path': args.image_path,
    'input_desc': args.text,
    'query': args.query
    }
    
    corrected_sample = corrector.correct(sample)
    print(corrected_sample['output'])
    with open('intermediate_view.json', 'w', encoding='utf-8') as file:
        json.dump(corrected_sample, file, ensure_ascii=False, indent=4)
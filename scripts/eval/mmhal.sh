export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p ./eval/mmhal/answers
python eval/mmhal/eval.py \
    --data_root ./data_dir/MMHal-Bench \
    --processor_path ckpts/Qwen-VL-Chat \
    --model_path $CKPT \
    --output_path ./eval/mmhal/answers/${TAG}.json \

mkdir -p ./eval/mmhal/results
python eval/mmhal/eval_gpt4.py \
    --response ./eval/mmhal/answers/${TAG}.json \
    --evaluation ./eval/mmhal/results/${TAG}.json \
    --api_info_path ./eval/mmhal/gpt4_0613_api.json \
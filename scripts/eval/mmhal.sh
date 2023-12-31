export PYTHONPATH=$PYTHONPATH:$PWD
python eval/mmhal/eval.py \
    --data_root ./data_dir/MMHal-Bench \
    --model_path $CKPT \
    --output_path ./eval/mmhal/answers/${TAG}.json \

python eval/mmhal/eval_gpt4.py \
    --response ./eval/mmhal/answers/${TAG}.json \
    --evaluation ./eval/mmhal/results/${TAG}.json \
    --api_info_path ./eval/mmhal/api.json \
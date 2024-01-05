export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p ./eval/mmbench/results/${TAG}
python eval/mmbench/eval.py \
    --data_root ./data_dir/mmbench_dev_en_20231003.tsv \
    --model_path $CKPT \
    --output_path ./eval/mmbench/results/${TAG}/${TAG}.xlsx \
    --batch_size 16 \
    --api_info_path ./eval/mmbench/gpt-35-turbo_api.json
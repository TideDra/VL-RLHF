export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p ./eval/mmbench/results
python eval/mmbench/eval.py \
    --data_root ./data_dir/mmbench_dev_en_20231003.tsv \
    --model_path $CKPT \
    --output_path ./eval/mmbench/results/${TAG}.xlsx \
    --batch_size 16 \
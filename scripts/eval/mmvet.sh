export PYTHONPATH=$PYTHONPATH:$PWD
python ./eval/mmvet/eval.py \
    --data_root ./data_dir/mm-vet \
    --model_path $CKPT \
    --output_path ./eval/mmvet/results/${TAG}.json \
    --batch_size 16
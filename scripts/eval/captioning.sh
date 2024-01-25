export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p ./eval/captioning/results
python eval/captioning/generate.py \
    --data_root ./data_dir/coco2017/test2017 \
    --processor_path ckpts/Qwen-VL-Chat \
    --model_path $CKPT \
    --output_path ./eval/captioning/results/${TAG}.json \
    --batch_size 16 \
    --sample_num 200
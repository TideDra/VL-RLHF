export PYTHONPATH=$PYTHONPATH:$PWD
batch_size=${BATCH_SIZE:-16}
mkdir -p ./eval/vqa/results
accelerate launch --config_file accelerate_config/infer.yaml \
    -m vlrlhf.eval.vqa.generate \
    --data_root ./data_dir/vg/VG_all \
    --file_path ./data_dir/vg/obj_existence_prompts.json \
    --model_path $CKPT \
    --output_path ./eval/vqa/results/${TAG}.json \
    --batch_size $batch_size \

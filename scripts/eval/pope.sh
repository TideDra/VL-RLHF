export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
gpu_number=$(nvidia-smi --list-gpus | wc -l)
batch_size=${BATCH_SIZE:-16}
mkdir -p ./eval/pope/results
accelerate launch --config_file accelerate_config/infer.yaml --num_processes $gpu_number \
    -m vlrlhf.eval.pope.eval \
    --image_root ./data_dir/coco2014/val2014 \
    --file_root ./data_dir/POPE/output/coco \
    --model_path $CKPT \
    --output_path ./eval/pope/results/${TAG}.json \
    --batch_size $batch_size \
    --report_to_mysql $report_to_mysql \
    --sql_host $SQL_HOST \
    --sql_port $SQL_PORT \
    --sql_user $SQL_USER \
    --sql_password $SQL_PASSWORD \
    --sql_db $SQL_DB \
    --sql_tag $TAG

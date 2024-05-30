export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
gpu_number=$(nvidia-smi --list-gpus | wc -l)
mkdir -p ./eval/pope/results

if [[ -e "${CKPT}/adapter_config.json" ]]; then
    if [[ ! -d "${CKPT}/merged" ]]; then
        python src/vlrlhf/merge_peft_model.py --adapter_path $CKPT
    fi
    model_path="${CKPT}/merged"
else
    model_path=$CKPT
fi

python -m vlrlhf.eval.pope.eval_sgl \
    --image_root ./data_dir/coco2014/val2014 \
    --file_root ./data_dir/POPE/output/coco \
    --model_path $model_path \
    --output_path ./eval/pope/results/${TAG}.json \
    --batch_size 128 \
    --report_to_mysql $report_to_mysql \
    --sql_host $SQL_HOST \
    --sql_port $SQL_PORT \
    --sql_user $SQL_USER \
    --sql_password $SQL_PASSWORD \
    --sql_db $SQL_DB \
    --sql_tag $TAG

if [[ -d "${CKPT}/merged" ]]; then
    rm -rf "${CKPT}/merged"
fi

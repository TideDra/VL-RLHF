export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
mkdir -p ./eval/seedbench/results
batch_size=${BATCH_SIZE:-16}
if [[ -e "${CKPT}/adapter_config.json" ]]; then
    if [[ ! -d "${CKPT}/merged" ]]; then
        python src/vlrlhf/merge_peft_model.py --adapter_path $CKPT
    fi
    model_path="${CKPT}/merged"
else
    model_path=$CKPT
fi


python -m vlrlhf.eval.seedbench.eval_generate_sgl \
    --data_root ./data_dir/SEED-Bench \
    --model_path $model_path \
    --output_path ./eval/seedbench/results/${TAG}.json \
    --batch_size 128

if [[ -d "${CKPT}/merged" ]]; then
    rm -rf "${CKPT}/merged"
fi

python -m vlrlhf.eval.seedbench.calculate \
        --result_file ./eval/seedbench/results/${TAG}.json \
        --anno_path ./data_dir/SEED-Bench/SEED-Bench.json \
        --report_to_mysql $report_to_mysql \
        --sql_host $SQL_HOST \
        --sql_port $SQL_PORT \
        --sql_user $SQL_USER \
        --sql_password $SQL_PASSWORD \
        --sql_db $SQL_DB \
        --sql_tag $TAG

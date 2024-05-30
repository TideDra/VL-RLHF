export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
mkdir -p ./eval/mmvet/results

if [[ -e "${CKPT}/adapter_config.json" ]]; then
    if [[ ! -d "${CKPT}/merged" ]]; then
        python src/vlrlhf/merge_peft_model.py --adapter_path $CKPT
    fi
    model_path="${CKPT}/merged"
else
    model_path=$CKPT
fi


python -m vlrlhf.eval.mmvet.eval_sgl \
    --data_root ./data_dir/mm-vet \
    --model_path $model_path \
    --output_path ./eval/mmvet/results/${TAG}.json \
    --batch_size 128

if [[ -d "${CKPT}/merged" ]]; then
    rm -rf "${CKPT}/merged"
fi

python -m vlrlhf.eval.mmvet.calculate \
        --result_file ./eval/mmvet/results/${TAG}.json \
        --report_to_mysql $report_to_mysql \
        --sql_host $SQL_HOST \
        --sql_port $SQL_PORT \
        --sql_user $SQL_USER \
        --sql_password $SQL_PASSWORD \
        --sql_db $SQL_DB \
        --sql_tag $TAG

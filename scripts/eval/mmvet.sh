export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
mkdir -p ./eval/mmvet/results
batch_size=${BATCH_SIZE:-16}
accelerate launch --config_file accelerate_config/infer.yaml \
    -m vlrlhf.eval.mmvet.eval \
    --data_root ./data_dir/mm-vet \
    --model_path $CKPT \
    --output_path ./eval/mmvet/results/${TAG}.json \
    --batch_size $batch_size

python -m vlrlhf.eval.mmvet.calculate \
        --result_file ./eval/mmvet/results/${TAG}.json \
        --report_to_mysql $report_to_mysql \
        --sql_host $SQL_HOST \
        --sql_port $SQL_PORT \
        --sql_user $SQL_USER \
        --sql_password $SQL_PASSWORD \
        --sql_db $SQL_DB \
        --sql_tag $TAG

export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
mkdir -p ./eval/seedbench/results
batch_size=${BATCH_SIZE:-16}
accelerate launch --config_file accelerate_config/infer.yaml \
    -m vlrlhf.eval.seedbench.eval \
    --data_root ./data_dir/SEED-Bench \
    --model_path $CKPT \
    --output_path ./eval/seedbench/results/${TAG}.json \
    --batch_size $batch_size

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

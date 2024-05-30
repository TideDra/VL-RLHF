export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
mkdir -p ./eval/seedbench/results
batch_size=${BATCH_SIZE:-16}
accelerate launch --config_file accelerate_config/infer.yaml \
    -m vlrlhf.eval.seedbench.eval_generate \
    --data_root ./data_dir/SEED-Bench \
    --model_path $CKPT \
    --output_path ./eval/seedbench/results/${TAG}.json \
    --batch_size $batch_size

eval "source ${conda_path} ${vlmeval_env_name}"

python  src/vlrlhf/eval/seedbench/extract_choice.py \
        --result_file ./eval/seedbench/results/${TAG}.json \
        --output_file ./eval/seedbench/results/${TAG}_choice.json \
        --model_path ckpts/Qwen1.5-14B-Chat

eval "source ${conda_path} ${vlrlhf_env_name}"

python -m vlrlhf.eval.seedbench.calculate \
        --result_file ./eval/seedbench/results/${TAG}_choice.json \
        --anno_path ./data_dir/SEED-Bench/SEED-Bench.json \
        --report_to_mysql $report_to_mysql \
        --sql_host $SQL_HOST \
        --sql_port $SQL_PORT \
        --sql_user $SQL_USER \
        --sql_password $SQL_PASSWORD \
        --sql_db $SQL_DB \
        --sql_tag $TAG

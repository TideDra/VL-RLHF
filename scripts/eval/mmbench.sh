export PYTHONPATH=$PYTHONPATH:$PWD
source scripts/eval/config.sh
gpu_number=$(nvidia-smi --list-gpus | wc -l)
DATA_DIR=data_dir/MMBench_DEV_EN.tsv
batch_size=${BATCH_SIZE:-16}
workdir=./eval/mmbench/results/${TAG}

# if workdir exists, remove it
if [ -d $workdir ]; then
    rm -rf $workdir
fi
mkdir -p $workdir
accelerate launch --config_file accelerate_config/infer.yaml --num_processes $gpu_number \
    -m vlrlhf.eval.mmbench.eval \
    --data_root $DATA_DIR \
    --model_path $CKPT \
    --output_path ${workdir}/${TAG}.xlsx \
    --batch_size $batch_size

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
fi

eval "source ${conda_path} ${vlmeval_env_name}"
lmdeploy serve api_server ${judger_path} --server-port ${judger_port} --tp ${judger_tp}  > ${workdir}/server.log 2>&1 &

# wait for 'Application startup complete.' appear in the log
while ! grep -q 'Application startup complete.' ${workdir}/server.log; do
    sleep 1
done

python -m vlmeval.evaluate.multiple_choice  ${workdir}/${TAG}.xlsx

eval "source ${conda_path} ${vlrlhf_env_name}"
python -m vlrlhf.eval.mmbench.calculate \
        --result_file ${workdir}/${TAG}.xlsx \
        --report_to_mysql $report_to_mysql \
        --sql_host $SQL_HOST \
        --sql_port $SQL_PORT \
        --sql_user $SQL_USER \
        --sql_password $SQL_PASSWORD \
        --sql_db $SQL_DB \
        --sql_tag $TAG

# kill the server
kill -INT $!

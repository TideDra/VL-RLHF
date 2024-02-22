export PYTHONPATH=$PYTHONPATH:$PWD
DATA_DIR=./data_dir/MME
mkdir -p ./eval/mme/answers
python -m vlrlhf.eval.mme.eval \
    --data_root $DATA_DIR \
    --processor_path ckpts/Qwen-VL-Chat \
    --model_path $CKPT \
    --output_path ./eval/mme/answers/${TAG}.jsonl \


python -m vlrlhf.eval.mme.convert_answer_to_mme --experiment ${TAG} --data-path $DATA_DIR

cd eval/mme
mkdir -p results
python -m vlrlhf.eval.mme.calculation --results_dir results/${TAG}
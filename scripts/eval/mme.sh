export PYTHONPATH=$PYTHONPATH:$PWD
DATA_DIR=./data_dir/MME
mkdir -p ./eval/mme/answers
python eval/mme/eval.py \
    --data_root $DATA_DIR \
    --model_path $CKPT \
    --output_path ./eval/mme/answers/${TAG}.jsonl \


python eval/mme/convert_answer_to_mme.py --experiment ${TAG} --data-path $DATA_DIR

cd eval/mme
mkdir -p results
python calculation.py --results_dir results/${TAG}
export PYTHONPATH=$PYTHONPATH:$PWD
python -m vlrlhf.eval.mmvet.eval_sgl \
    --data_root ./data_dir/mm-vet \
    --output_path ${TAG}.json \
    --batch_size 16
export PYTHONPATH=$PYTHONPATH:$PWD
adapter_name="bs_128_ep_3_mg_-1_bt__lr_1e-5"
export WANDB_MODE="online"
python eval/eval_rm.py \
    --data_root ./data_dir/MMHal-Bench \
    --model_path ckpts/Qwen-VL-Chat \
    --reward_adapter "ckpts/Qwen-VL-Chat-rm/${adapter_name}" \
    --best_of 32 \
    --temperature 1.0 \
    --project_name "VL-RLHF" \
    --group_name "Qwen-VL-rm-eval" \
    --run_name $adapter_name \

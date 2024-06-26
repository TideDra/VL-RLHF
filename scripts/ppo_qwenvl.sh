per_device_batch_size=64
mini_batch_size=2
gradient_accumulation_steps=8
backward_batch_size=$((mini_batch_size * gradient_accumulation_steps))
ppo_epochs=4
lr=5e-6
vf_coef=1
init_kl_coef=0.3
horizon=5000
gpu_number=$(nvidia-smi --list-gpus | wc -l)
global_backward_bs=$((backward_batch_size * gpu_number))
global_bs=$((per_device_batch_size * gpu_number))
vinit=0.1
name="bs_${global_bs}_bbs_${global_backward_bs}_ppoep_${ppo_epochs}_lr_${lr}_fixkl_${init_kl_coef}_vfcoef_${vf_coef}_vinit_${vinit}_vlfeedback60k_rm20k_mg-1"
export WANDB_MODE="online"
accelerate launch --config_file accelerate_config/ddp.yaml --num_processes $gpu_number\
        src/vlrlhf/ppo.py \
        --model_name_or_path ckpts/Qwen-VL-Chat \
        --output_dir ckpts/Qwen-VL-Chat-ppo/$name \
        --data_dir data_dir/VLQueryData/vlfeedback_60k.json \
        --image_root data_dir/ \
        --remove_unused_columns False \
        --reward_adapter ckpts/Qwen-VL-Chat-rm/bs_128_ep_3_mg_0_bt__lr_1e-5_vlfeedback20k \
        --use_lora True \
        --use_value_adapter True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules "c_attn,attn.c_proj,w1,w2" \
        --lora_bias "none" \
        --batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $lr \
        --gradient_checkpointing False \
        --bf16 True \
        --tf32 True \
        --mini_batch_size $mini_batch_size \
        --ppo_epochs $ppo_epochs \
        --adap_kl_ctrl False \
        --init_kl_coef $init_kl_coef \
        --horizon $horizon \
        --max_new_tokens 512 \
        --vf_coef $vf_coef \
        --v_head_initializer_range $vinit \
        --whiten_rewards True \
        --log_with wandb \
        --run_name  $name \
        --project_name "VL-RLHF" \
        --group_name "Qwen-VL-Chat-ppo" \
        --per_device_gamelog_size 2

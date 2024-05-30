source scripts/config.sh

per_device_train_batch_size=16
gradient_accumulation_steps=2
epoch=1
margin=-1
beta=0.1
lr=1e-5
dr=1.0
lm_lora_modules="c_attn,attn.c_proj,w1,w2"
vision_lora_modules="in_proj,out_proj,c_fc"
full_lora_modules="${lm_lora_modules},${vision_lora_modules}"
gpu_number=$(nvidia-smi --list-gpus | wc -l)
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
dataset="qwenvl_vg30khal_paired"
loss_type="ddpo"
name="bs${global_bs}_ep${epoch}_mg${margin}_bt${beta}_lr${lr}_${dataset}_dr${dr}"
accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number\
        src/vlrlhf/dpo.py \
        --model_name_or_path ckpts/Qwen-VL-Chat \
        --output_dir ckpts/Qwen-VL-Chat-ddpo/$name \
        --data_path ${dataset_map[$dataset]} \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_ratio $dr \
        --image_root ${image_root_map[$dataset]} \
        --freeze_vision_tower True \
        --use_flash_attention_2 False \
        --use_lora True \
        --loss_type $loss_type \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules $lm_lora_modules \
        --lora_bias "none" \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_train_epochs $epoch \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate $lr \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --score_margin $margin \
        --remove_unused_columns False \
        --beta $beta \
        --max_length 2048 \
        --max_prompt_length 1024 \
        --max_target_length 512 \
        --eval_strategy "steps" \
        --eval_steps 200 \
        --save_strategy "steps" \
        --save_steps 30 \
        --save_total_limit 10 \
        --logging_first_step False \
        --logging_steps 10 \
        --report_to wandb \
        --run_name  $name\
        --project_name "VL-RLHF" \
        --group_name "Qwen-VL-Chat-ddpo" \

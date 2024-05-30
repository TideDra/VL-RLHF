source scripts/config.sh

per_device_train_batch_size=16
gradient_accumulation_steps=1
epoch=1
lr=5e-7
lm_lora_modules="auto"
dataset="instructblip_vg30k_conv"
gpu_number=$(nvidia-smi --list-gpus | wc -l)
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
name="bs${global_bs}_ep${epoch}_lr_${lr}_${dataset}"

accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number\
        src/vlrlhf/sft.py \
        --model_name_or_path ckpts/instructblip-vicuna-13b \
        --output_dir ckpts/instructblip-vicuna-13b-sft/$name \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_path ${dataset_map[$dataset]} \
        --image_root ${image_root_map[$dataset]} \
        --dataset_num_proc 16 \
        --freeze_vision_tower True \
        --use_lora True \
        --use_flash_attention_2 True \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --lora_bias none \
        --lora_target_modules $lm_lora_modules \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_train_epochs $epoch \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate $lr \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --remove_unused_columns False \
        --max_length 2048 \
        --eval_strategy "steps" \
        --eval_steps 200 \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 10 \
        --logging_first_step False \
        --logging_steps 10 \
        --report_to wandb \
        --run_name  $name\
        --project_name "VL-RLHF" \
        --group_name "instructblip-vicuna-13b-sft" \

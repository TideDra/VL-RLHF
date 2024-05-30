source scripts/config.sh


per_device_train_batch_size=16
gradient_accumulation_steps=1
epoch=1
margin=-1
beta=0.1
lr=1e-6
lm_lora_modules="auto"
dataset="llava13b_vg30k_paired"
gpu_number=$(nvidia-smi --list-gpus | wc -l)
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
name="bs${global_bs}_ep${epoch}_mg${margin}_bt${beta}_lr${lr}_${dataset}"
loss_type="kto_pair"
accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number\
        src/vlrlhf/dpo.py \
        --model_name_or_path ckpts/llava-1.5-13b-hf \
        --output_dir ckpts/llava-1.5-13b-kto/$name \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_path ${dataset_map[$dataset]} \
        --image_root ${image_root_map[$dataset]} \
        --freeze_vision_tower True \
        --use_lora True \
        --loss_type $loss_type \
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
        --score_margin $margin \
        --remove_unused_columns False \
        --beta $beta \
        --max_length 2048 \
        --max_prompt_length 1024 \
        --max_target_length 512 \
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
        --group_name "llava-1.5-13b-kto" \

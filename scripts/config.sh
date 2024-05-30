declare -A dataset_map=(
        ["rlhfv"]="HaoyeZhang/RLHF-V-Dataset"
        ["VLFeedback"]="data_dir/VLFeedback"
        ["ultrafeedback"]="data_dir/ultrafeedback_cleaned.json"
        ["rlaifv"]="data_dir/rlaif_v.json"
)

declare -A dataset_name_map=(
        ["rlhfv"]="rlhfv"
        ["VLFeedback"]="vlfeedback_paired"
        ["ultrafeedback"]="plain_dpo"
        ["rlaifv"]="plain_dpo"
)

declare -A image_root_map=(
        ["rlhfv"]="data_dir/RLHF-V"
        ["VLFeedback"]="data_dir/VLFeedback"
        ["ultrafeedback"]="None"
        ["rlaifv"]="data_dir/RLAIF-V"
)

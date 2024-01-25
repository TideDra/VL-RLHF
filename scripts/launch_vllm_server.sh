export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --download-dir ckpts \
    --dtype "auto" \
    --tensor-parallel-size 4
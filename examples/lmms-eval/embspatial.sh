# Tested with NVIDIA-SMI 550.90.07 | Driver Version: 550.90.07 | CUDA Version: 13.1
# Recommended to use "torch==2.7.1", "torchvision==0.22.1"
# Installation
# uv venv -p 3.11
# source .venv/bin/activate
# uv pip install ./lmms-eval spacy
# uv pip install flash-attn --no-build-isolation

# Qwen/Qwen3-VL-4B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-4B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks embspatial \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-8B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks embspatial \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
# Tested with NVIDIA-SMI 550.90.07 | Driver Version: 550.90.07 | CUDA Version: 13.1
# Recommended to use "torch==2.7.1", "torchvision==0.22.1"
# Installation
# uv venv -p 3.11
# source .venv/bin/activate
# uv pip install ./lmms-eval spacy
# uv pip install flash-attn --no-build-isolation

# InternVL2-OpenGVLab/InternVL3-14B
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model internvl2 \
    --model_args=pretrained=OpenGVLab/InternVL3-14B \
    --tasks viewspatial \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-3B-Instruct
MIN_PIXELS=$((256*28*28))
MAX_PIXELS=$((1280*28*28))
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks viewspatial \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-7B-Instruct
MIN_PIXELS=$((256*28*28))
MAX_PIXELS=$((1280*28*28))
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks viewspatial \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
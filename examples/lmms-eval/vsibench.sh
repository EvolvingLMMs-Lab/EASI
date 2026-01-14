# Tested with NVIDIA-SMI 550.90.07 | Driver Version: 550.90.07 | CUDA Version: 13.1
# Recommended to use "torch==2.7.1", "torchvision==0.22.1"
# Installation
# uv venv -p 3.11
# source .venv/bin/activate
# uv pip install ./lmms-eval spacy
# uv pip install flash-attn --no-build-isolation

# InternVL3.5-8B for VSiBench
NUM_FRAMES=128
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model internvl3_5 \
    --model_args=pretrained=OpenGVLab/InternVL3_5-8B,num_frame=${NUM_FRAMES},modality="video" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# nyu-visionx/Cambrian-S-3B for VSiBench
# uv pip install git+https://github.com/cambrian-mllm/cambrian-s.git
NUM_FRAMES=32
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12345 \
    -m lmms_eval \
    --model cambrians \
    --model_args=pretrained=nyu-visionx/Cambrian-S-3B,conv_template=qwen_2,video_max_frames=${NUM_FRAMES},miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN} \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# nyu-visionx/Cambrian-S-7B for VSiBench
# uv pip install git+https://github.com/cambrian-mllm/cambrian-s.git
NUM_FRAMES=32
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12345 \
    -m lmms_eval \
    --model cambrians \
    --model_args=pretrained=nyu-visionx/Cambrian-S-7B,conv_template=qwen_2,video_max_frames=${NUM_FRAMES},miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN} \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-3B-Instruct for VSiBench
NUM_FRAMES=128
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,max_num_frames=${NUM_FRAMES},attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-7B-Instruct for VSiBench
NUM_FRAMES=128
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=${NUM_FRAMES},attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/


# SenseNova/SenseNova-SI-1.1-Qwen3-VL-8B for VSiBench with multi-image input
# This command uses multi-image input (video frames as images) instead of native video input
# The number of frames is controlled by num_frames in the task YAML (default: 32)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=sensenova/SenseNova-SI-1.1-Qwen3-VL-8B,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs "max_new_tokens=2048,temperature=0,top_p=1.0,num_beams=1,do_sample=false" \
    --tasks vsibench_multiimage \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# SenseNova/SenseNova-SI-1.1-Qwen2.5-VL-3B for VSiBench with multi-image input
# This command uses multi-image input (video frames as images) instead of native video input
# The number of frames is controlled by num_frames in the task YAML (default: 32
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs "max_new_tokens=2048,temperature=0,top_p=1.0,num_beams=1,do_sample=false" \
    --tasks vsibench_multiimage \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# SenseNova/SenseNova-SI-1.1-Qwen2.5-VL-7B for VSiBench with multi-image input
# This command uses multi-image input (video frames as images) instead of native video input
# The number of frames is controlled by num_frames in the task YAML (default: 32
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs "max_new_tokens=2048,temperature=0,top_p=1.0,num_beams=1,do_sample=false" \
    --tasks vsibench_multiimage \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# SenseNova/SenseNova-SI-1.2-InternVL3-8B for VSiBench
NUM_FRAMES=32
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model internvl3 \
    --model_args=pretrained=sensenova/SenseNova-SI-1.2-InternVL3-8B,num_frame=${NUM_FRAMES},modality="video" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
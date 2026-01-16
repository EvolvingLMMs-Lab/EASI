# Tested with NVIDIA-SMI 550.90.07 | Driver Version: 550.90.07 | CUDA Version: 13.1
# Recommended to use "torch==2.7.1", "torchvision==0.22.1"
# Installation
# uv venv -p 3.11
# source .venv/bin/activate
# uv pip install ./lmms-eval spacy
# uv pip install flash-attn --no-build-isolation

# sensenova/SenseNova-SI-1.3-InternVL3-8B for SiteBench Image
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model internvl3 \
    --model_args=pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B \
    --tasks site_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# sensenova/SenseNova-SI-1.3-InternVL3-8B for SiteBench Video
NUM_FRAME=32
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model internvl3 \
    --model_args=pretrained=sensenova/SenseNova-SI-1.2-InternVL3-8B,num_frame=${NUM_FRAME},modality="video" \
    --tasks site_bench_video \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-3B-Instruct for SiteBench Image
# align the min and max pixels with vlmevalkit: https://github.com/EvolvingLMMs-Lab/VLMEvalKit/blob/1e8d17cd4969e129ef2ab81b077ea9627d464376/vlmeval/config.py#L1571C5-L1577C7
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((16384*28*28))
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2,use_custom_video_loader=True \
    --tasks site_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# Qwen/Qwen2.5-VL-3B-Instruct for SiteBench Video
NUM_FRAMES=32
FPS=2
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,fps=${FPS},max_num_frames=${NUM_FRAMES},attn_implementation=flash_attention_2,use_custom_video_loader=True \
    --tasks site_bench_video \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

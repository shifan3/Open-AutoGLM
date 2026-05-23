DOCKER_VERSION=v0.5.12 \
MODEL_PATH=/mnt/data5/pretrained_local/llm/ZhipuAI/AutoGLM-Phone-9B-Multilingual \
CONTAINER_NAME=autoglm-phone-9b-multilingual-sglang-backends \
SERVED_MODEL_NAME=autoglm-phone \
CUDA_VISIBLE_DEVICES=2 \
MAX_MODEL_LEN=65536 \
BASE_PORT=8101 \
bash start_sglang.sh
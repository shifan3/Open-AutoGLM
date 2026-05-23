#!/bin/bash



export BASE_PORT=${BASE_PORT:-8001}
export SGLANG_DOCKER_VERSION=${DOCKER_VERSION:-v0.5.7}
export MODEL_PATH=$MODEL_PATH
export DRAFT_MODEL_PATH=$DRAFT_MODEL_PATH

export NETWORK=${NETWORK:-host}
export SGLANG_CONTAINER_NAME=${CONTAINER_NAME:-sglang-worker}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-chat}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
export PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
export EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS:-""}

if [ -z $MODEL_PATH ]; then
    echo "MODEL_PATH is not set"
    exit 1
fi

if [ -z $SERVED_MODEL_NAME ]; then
    echo "SERVED_MODEL_NAME is not set"
    exit 1
fi

GREEN='\033[1;32m'
GRAY='\033[0;37m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NOCOLOR='\033[0m'

IMAGE="lmsysorg/sglang:$SGLANG_DOCKER_VERSION"



if [ ! -z "$DRAFT_MODEL_PATH" ]; then
    EXTRA_SGLANG_ARGS=" --speculative-draft-model $DRAFT_MODEL_PATH --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 $EXTRA_SGLANG_ARGS"
fi

if [ ! -z "$MEM_FRACTION_STATIC" ]; then
    EXTRA_SGLANG_ARGS=" --mem-fraction-static $MEM_FRACTION_STATIC $EXTRA_SGLANG_ARGS"
fi


if [ -z $CUDA_VISIBLE_DEVICES ]; then
    export NUM_GPUS=`nvidia-smi --query-gpu=name --format=csv,noheader | grep NVIDIA | wc -l`
    export CUDA_VISIBLE_DEVICES=`seq -s, $counter 0 $(($NUM_GPUS-1))`
else
    export NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}'`
fi
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_WORKER=$((NUM_GPUS / PIPELINE_PARALLEL_SIZE / TENSOR_PARALLEL_SIZE))
echo "NUM_WORKER: $NUM_WORKER"

if [[ "${MODEL_PATH:0:1}" != "/" ]]; then
    MODEL_PATH=`pwd`/$MODEL_PATH
fi

echo "Launching $NUM_WORKER workers using ${NUM_GPUS} GPU(s): ${GPUS[*]}"
mkdir -p logs
docker ps -a -q --filter "name=$SGLANG_CONTAINER_NAME-" | xargs -r docker rm -f
for ((i=0; i<NUM_WORKER; i++)); do
    START_IDX=$((i * PIPELINE_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE))
    END_IDX=$(( (i + 1) * PIPELINE_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE - 1 ))
    GPU_INDEXES=()
    for ((j=START_IDX; j<=END_IDX; j++)); do
        GPU_INDEXES+=("${GPUS[$j]}")
    done
    GPU=$(IFS=,; echo "${GPU_INDEXES[*]}")
    if [ $NETWORK == "host" ]; then
        PORT=$((BASE_PORT + i))
    else
        PORT=$BASE_PORT
    fi
    CONTAINER_NAME="$SGLANG_CONTAINER_NAME-$i"
    LOG_FILE="logs/sglang-${CONTAINER_NAME}.log"

    docker kill "$CONTAINER_NAME" > /dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
    rm -f $LOG_FILE
    touch $LOG_FILE
    echo "Starting container $CONTAINER_NAME on GPU $GPU (port $PORT)"
    echo "EXTRA_SGLANG_ARGS: $EXTRA_SGLANG_ARGS"

    docker run -d \
        --runtime=nvidia \
        --gpus=all \
        --name "$CONTAINER_NAME" \
        --ipc=host \
        --net=$NETWORK \
        --shm-size 20g \
        -v "$(pwd):/root/engine" \
        -v $MODEL_PATH:$MODEL_PATH \
        -e CUDA_VISIBLE_DEVICES=$GPU \
        $IMAGE python -m sglang.launch_server \
        --host 0.0.0.0 \
        --port $PORT \
        --model-path $MODEL_PATH \
        --served-model-name $SERVED_MODEL_NAME \
        --dtype bfloat16 \
        --enable-multimodal \
        --max-running-requests 128 \
        --context-length $MAX_MODEL_LEN \
        --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --data-parallel-size 1 \
        --attention-backend flashinfer \
        --limit-mm-data-per-request  '{"image": 2, "video": 0, "audio": 0}' \
        --enable-metrics \
        $EXTRA_SGLANG_ARGS 
        #--chunked-prefill-size 40 \ ?

    docker logs -f "$CONTAINER_NAME" > $LOG_FILE 2>&1 &
done



echo "Starting health checks..."



for ((i=0; i<NUM_WORKER; i++)); do
  SGLANG_STARTED=0
  CONTAINER_NAME="$SGLANG_CONTAINER_NAME-$i"
  LOG_FILE="logs/sglang-${CONTAINER_NAME}.log"
  echo -e "${GREEN}wait sglang: $CONTAINER_NAME started ${NOCOLOR}"
  PREV_LINE=

  for i in {1..60000} #wait one hour
  do
      sleep 0.1
      if [ ! -f $LOG_FILE ]; then
          continue
      fi
      if ! docker ps | grep $CONTAINER_NAME > /dev/null 2>&1; then
        echo -e "${RED}sglang: $CONTAINER_NAME not running, abort${NOCOLOR}"
        break
    fi
      if grep "Application startup complete" $LOG_FILE ; then
          SGLANG_STARTED=1
          break
      else
          CURR_LINE=`cat $LOG_FILE | tail -n 1`

          if [ "$CURR_LINE" != "$PREV_LINE" ]; then
              echo -e "${GRAY}$CURR_LINE${NOCOLOR}"
              PREV_LINE=$CURR_LINE
          fi
      fi
  done

  if [ $SGLANG_STARTED -eq 0 ]; then
      echo -e "${RED}sglang: $CONTAINER_NAME not ready, abort${NOCOLOR}"
      exit 1
  fi


done

echo "✅ All workers are healthy!"
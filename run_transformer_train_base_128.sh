#!/bin/bash

# semantic label indexing
DATASET=$1
LABEL_EMB=pifa
ALGO=5
SEED=0

GPUS=2,3,4,5
#GPUS=6,7,8,9

MODEL_TYPE=$2
MODEL_NAME=$3

if [ ${DATASET} == "Eurlex-4K" ]; then
  PER_GPU_TRN_BSZ=16
  PER_GPU_VAL_BSZ=128
  MAX_STEPS_ARR=( 4000 6000 8000 )
  WARMUP_STEPS_ARR=( 400 600 800 )
  LOGGING_STEPS=100
  SAVE_STEPS=400
elif [ ${DATASET} == "Wiki10-31K" ]; then
  PER_GPU_TRN_BSZ=16
  PER_GPU_VAL_BSZ=128
  MAX_STEPS_ARR=( 4000 6000 8000 )
  WARMUP_STEPS_ARR=( 400 600 800 )
  LOGGING_STEPS=100
  SAVE_STEPS=400
elif [ ${DATASET} == "AmazonCat-13K" ]; then
  PER_GPU_TRN_BSZ=32
  PER_GPU_VAL_BSZ=128
  MAX_STEPS_ARR=( 16000 24000 )
  WARMUP_STEPS_ARR=( 1600 2400 )
  LOGGING_STEPS=500
  SAVE_STEPS=4000
elif [ ${DATASET} == "Wiki-500K" ]; then
  PER_GPU_TRN_BSZ=32
  PER_GPU_VAL_BSZ=128
  MAX_STEPS_ARR=( 65000 )
  WARMUP_STEPS_ARR=( 6500 )
  LOGGING_STEPS=1000
  SAVE_STEPS=10000
else
  echo "dataset not support [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
  exit
fi

# grid hyper-params
MAX_XSEQ_LEN=128
GRAD_ACCU_STEPS=1
#LEARNING_RATE_ARR=( 1e-5 2e-5 3e-5 )
LEARNING_RATE_ARR=( 5e-5 8e-5 1e-4 )


for idx in "${!MAX_STEPS_ARR[@]}"; do
  MAX_STEPS=${MAX_STEPS_ARR[${idx}]}
  WARMUP_STEPS=${WARMUP_STEPS_ARR[${idx}]}
  for LEARNING_RATE in "${LEARNING_RATE_ARR[@]}"; do
    # train transformer models on XMC preprocessed data
    OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
    MODEL_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/step-${MAX_STEPS}_warmup-${WARMUP_STEPS}_lr-${LEARNING_RATE}
    mkdir -p ${MODEL_DIR}
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.transformer \
      -m ${MODEL_TYPE} -n ${MODEL_NAME} \
      -i ${OUTPUT_DIR}/data-bin/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/data_dict.pt \
      -o ${MODEL_DIR} --overwrite_output_dir \
      --do_train --do_eval --stop_by_dev \
      --per_gpu_train_batch_size ${PER_GPU_TRN_BSZ} \
      --per_gpu_eval_batch_size ${PER_GPU_VAL_BSZ} \
      --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
      --max_steps ${MAX_STEPS} \
      --warmup_steps ${WARMUP_STEPS} \
      --learning_rate ${LEARNING_RATE} \
      --logging_steps ${LOGGING_STEPS} \
      --save_steps ${SAVE_STEPS} \
      |& tee ${MODEL_DIR}/log.txt
    #exit
  done
done

#!/bin/bash

GPUS=$1
DATASET=$2
MODEL_TYPE=$3
MODEL_NAME=$4

INDEXER_NAME=pifa-a5-s0
#INDEXER_NAME=pifa-neural-a5-s0
#INDEXER_NAME=text-emb-a5-s0
OUTPUT_DIR=save_models/${DATASET}/${INDEXER_NAME}
MAX_XSEQ_LEN=128

# fp32 on V100
PER_GPU_TRN_BSZ=16
PER_GPU_VAL_BSZ=64

# set hyper-params by dataset
if [ ${DATASET} == "Eurlex-4K" ]; then
  #MAX_STEPS_ARR=( 1000 2000 )
  #WARMUP_STEPS_ARR=( 100 200 )
  MAX_STEPS_ARR=( 1000 )
  WARMUP_STEPS_ARR=( 100 )
  LOGGING_STEPS=20
  SAVE_STEPS=100
  GRAD_ACCU_STEPS=4
  LEARNING_RATE_ARR=( 7e-5 )
elif [ ${DATASET} == "Wiki10-31K" ]; then
  #MAX_STEPS_ARR=( 2000 3000 )
  #WARMUP_STEPS_ARR=( 200 300 )
  MAX_STEPS_ARR=( 2000 )
  WARMUP_STEPS_ARR=( 200 )
  LOGGING_STEPS=20
  SAVE_STEPS=100
  GRAD_ACCU_STEPS=4
  LEARNING_RATE_ARR=( 5e-5 )
elif [ ${DATASET} == "AmazonCat-13K" ]; then
  MAX_STEPS_ARR=( 20000 )
  WARMUP_STEPS_ARR=( 2000 )
  LOGGING_STEPS=100
  SAVE_STEPS=2000
  GRAD_ACCU_STEPS=4
  LEARNING_RATE_ARR=( 1e-4 )
elif [ ${DATASET} == "Wiki-500K" ]; then
  MAX_STEPS_ARR=( 40000 )
  WARMUP_STEPS_ARR=( 4000 )
  LOGGING_STEPS=100
  SAVE_STEPS=2000
  GRAD_ACCU_STEPS=4
  LEARNING_RATE_ARR=( 8e-5 )
else
  echo "dataset not support [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
  exit
fi

for idx in "${!MAX_STEPS_ARR[@]}"; do
  MAX_STEPS=${MAX_STEPS_ARR[${idx}]}
  WARMUP_STEPS=${WARMUP_STEPS_ARR[${idx}]}
  for LEARNING_RATE in "${LEARNING_RATE_ARR[@]}"; do
    # train transformer models on XMC preprocessed data
    MODEL_DIR=${OUTPUT_DIR}/matcher-cased_fp32/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/step-${MAX_STEPS}_warmup-${WARMUP_STEPS}_lr-${LEARNING_RATE}
    mkdir -p ${MODEL_DIR}
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.transformer \
      -m ${MODEL_TYPE} -n ${MODEL_NAME} \
      -i ${OUTPUT_DIR}/data-bin-cased/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/data_dict.pt \
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
  done
done

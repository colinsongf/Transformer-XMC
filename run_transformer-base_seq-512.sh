#!/bin/bash

# semantic label indexing
GPUS=$1
DATASET=$2
LABEL_EMB=pifa
ALGO=5
SEED=0

MODEL_TYPE=$3
MODEL_NAME=$4
MAX_XSEQ_LEN=512
OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}

# set per_gpu_bsz by model_type
if [ ${MODEL_TYPE} == "bert" ] || [ ${MODEL_TYPE} == "roberta" ]; then
  PER_GPU_TRN_BSZ=16
  PER_GPU_VAL_BSZ=64
  GRAD_ACCU_STEPS=1
elif [ ${MODEL_TYPE} == "xlnet" ]; then
  PER_GPU_TRN_BSZ=8
  PER_GPU_VAL_BSZ=64
  GRAD_ACCU_STEPS=2
else
  echo "model_type not support [ bert | roberta | xlnet ]"
  exit
fi

# set hyper-params by dataset
if [ ${DATASET} == "Eurlex-4K" ]; then
  MAX_STEPS_ARR=( 1000 2000 )
  WARMUP_STEPS_ARR=( 100 200 )
  LOGGING_STEPS=50
  SAVE_STEPS=200
  LEARNING_RATE_ARR=( 1e-5 2e-5 3e-5 )
  INP_MODEL_DIR=${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}_seq-128/step-2000_warmup-200_lr-3e-5
elif [ ${DATASET} == "Wiki10-31K" ]; then
  MAX_STEPS_ARR=( 1000 2000 )
  WARMUP_STEPS_ARR=( 100 200 )
  LOGGING_STEPS=50
  SAVE_STEPS=200
  LEARNING_RATE_ARR=( 1e-5 2e-5 3e-5 )
  INP_MODEL_DIR=${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}_seq-128/step-2000_warmup-200_lr-3e-5
elif [ ${DATASET} == "AmazonCat-13K" ]; then
  MAX_STEPS_ARR=( 10000 )
  WARMUP_STEPS_ARR=( 1000 )
  LOGGING_STEPS=100
  SAVE_STEPS=1000
  LEARNING_RATE_ARR=( 3e-5 )
  INP_MODEL_DIR=${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}_seq-128/step-20000_warmup-2000_lr-1e-4
elif [ ${DATASET} == "Wiki-500K" ]; then
  MAX_STEPS_ARR=( 10000 )
  WARMUP_STEPS_ARR=( 1000 )
  LOGGING_STEPS=100
  SAVE_STEPS=1000
  LEARNING_RATE_ARR=( 3e-5 )
  INP_MODEL_DIR=${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}_seq-128/step-40000_warmup-4000_lr-1e-4
else
  echo "dataset not support [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
  exit
fi


for idx in "${!MAX_STEPS_ARR[@]}"; do
  MAX_STEPS=${MAX_STEPS_ARR[${idx}]}
  WARMUP_STEPS=${WARMUP_STEPS_ARR[${idx}]}
  for LEARNING_RATE in "${LEARNING_RATE_ARR[@]}"; do
    # train transformer models on XMC preprocessed data
    OUT_MODEL_DIR=${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/step-${MAX_STEPS}_warmup-${WARMUP_STEPS}_lr-${LEARNING_RATE}
    mkdir -p ${OUT_MODEL_DIR}
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m xbert.matcher.transformer \
      -m ${MODEL_TYPE} -n ${INP_MODEL_DIR} \
      -i ${OUTPUT_DIR}/data-bin-cased/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}/data_dict.pt \
      -o ${OUT_MODEL_DIR} --overwrite_output_dir \
      --do_train --do_eval --stop_by_dev --fp16 \
      --per_gpu_train_batch_size ${PER_GPU_TRN_BSZ} \
      --per_gpu_eval_batch_size ${PER_GPU_VAL_BSZ} \
      --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
      --max_steps ${MAX_STEPS} \
      --warmup_steps ${WARMUP_STEPS} \
      --learning_rate ${LEARNING_RATE} \
      --logging_steps ${LOGGING_STEPS} \
      --save_steps ${SAVE_STEPS} \
      |& tee ${OUT_MODEL_DIR}/log.txt
  done
done

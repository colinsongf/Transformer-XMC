#!/bin/bash

# set dataset and max_xseq_len
if [ $1 == 'Eurlex-4K' ]; then
	DATASET=Eurlex-4K
	DEPTH=6
elif [ $1 == 'Wiki10-31K' ]; then
	DATASET=Wiki10-31K
	DEPTH=9
elif [ $1 == 'AmazonCat-13K' ]; then
	DATASET=AmazonCat-13K
	DEPTH=8
elif [ $1 == 'Wiki-500K' ]; then
	DATASET=Wiki-500K
	DEPTH=13
else
	echo "unknown dataset for the experiment!"
	exit
fi
MAX_XSEQ_LEN=$2

# semantic label indexing
INDEXER_NAME=pifa-a5-s0
INDEXER_NAME=pifa-neural-a5-s0
INDEXER_NAME=text-emb-a5-s0

# HuggingFace pretrained model preprocess
MODEL_TYPE_ARR=( "bert" "roberta" "xlnet")
MODEL_NAME_ARR=( "bert-base-cased" "roberta-base" "xlnet-base-cased" )
#MODEL_NAME_ARR=( "bert-large-cased-whole-word-masking" "roberta-large" "xlnet-large-cased" )

for idx in "${!MODEL_TYPE_ARR[@]}"; do
	MODEL_TYPE=${MODEL_TYPE_ARR[${idx}]}
	MODEL_NAME=${MODEL_NAME_ARR[${idx}]}

  # semantic label indexing
	OUTPUT_DIR=save_models/${DATASET}/${INDEXER_NAME}
	INDEXER_DIR=${OUTPUT_DIR}/indexer

  # preprocess data binary (uncased) for transformer models
  DATABIN_DIR=${OUTPUT_DIR}/data-bin-cased/${MODEL_NAME}_seq-${MAX_XSEQ_LEN}
	mkdir -p ${DATABIN_DIR}
	python -u -m xbert.preprocess \
			-m ${MODEL_TYPE} \
      -n ${MODEL_NAME} \
			-i datasets/${DATASET} \
			-c ${INDEXER_DIR}/code.npz \
      --max_xseq_len $MAX_XSEQ_LEN \
			-o ${DATABIN_DIR} |& tee ${DATABIN_DIR}/log.txt

done

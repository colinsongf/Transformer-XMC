#!/bin/bash

# set DEPTH
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
elif [ $1 == 'Amazon-670K' ]; then
	DATASET=Amazon-670K
	DEPTH=13
else
	echo "unknown dataset for the experiment!"
	exit
fi

LABEL_NAME=pifa-a5-s0
OUTPUT_DIR=save_models_01-10/${DATASET}/${LABEL_NAME}

EXP_NAME=feat-joint_neg-yes
PRED_NPZ_PATHS=""
MODEL_NAME_ARR=( bert-base-cased_seq-128 roberta-base_seq-128 xlnet-base-cased_seq-128 bert-large-cased-whole-word-masking_seq-128 roberta-large_seq-128 )
for idx in "${!MODEL_NAME_ARR[@]}"; do
	MODEL_NAME=${MODEL_NAME_ARR[$idx]}

  # feature=tfidf, negative_sampling=no
  RANKER_DIR=${OUTPUT_DIR}/ranker_${EXP_NAME}
	mkdir -p ${RANKER_DIR}
  python -m xbert.ranker train \
    -x datasets/${DATASET}/X.trn.npz \
    -x2 ${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}/final_model/trn_embeddings.npy \
    -y datasets/${DATASET}/Y.trn.npz \
    -c ${OUTPUT_DIR}/indexer/code.npz \
    -z ${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}/final_model/C_trn_pred.npz \
    -o ${RANKER_DIR} -t 0.01

  PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.${MODEL_NAME}.npz
  python -m xbert.ranker predict \
    -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
    -x datasets/${DATASET}/X.tst.npz \
    -x2 ${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}/final_model/tst_embeddings.npy \
    -y datasets/${DATASET}/Y.tst.npz \
    -c ${OUTPUT_DIR}/matcher-cased/${MODEL_NAME}/final_model/C_tst_pred.npz

  PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
done

# final eval
EVAL_DIR=results_01-10/${DATASET}
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator -y datasets/${DATASET}/Y.tst.npz -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${EXP_NAME}.txt

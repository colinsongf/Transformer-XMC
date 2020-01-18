#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
	DEPTH=6
elif [ $1 == 'Wiki10-31K' ]; then
	DEPTH=9
elif [ $1 == 'AmazonCat-13K' ]; then
	DEPTH=8
elif [ $1 == 'Wiki-500K' ]; then
	DEPTH=13
elif [ $1 == 'Amazon-670K' ]; then
	DEPTH=13
elif [ $1 == 'Amazon-3M' ]; then
	DEPTH=15
else
	echo "unknown dataset for the experiment!"
	exit
fi

DATASET=$1
ALGO_LIST=( 5 )
SEED_LIST=( 0 1 2 )

#LABEL_EMB_LIST=( pifa )
#LABEL_EMB_LIST=( pifa-neural )
LABEL_EMB_LIST=( text-emb )

#EXP_NAME=pifa
EXP_NAME=pifa-neural
#EXP_NAME=text-emb
PRED_NPZ_PATHS=""
for idx in "${!LABEL_EMB_LIST[@]}"; do
	ALGO=${ALGO_LIST[$idx]}
	LABEL_EMB=${LABEL_EMB_LIST[$idx]}
	for SEED in "${SEED_LIST[@]}"; do
		# indexer
		OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}
		mkdir -p $OUTPUT_DIR/indexer
		python -m xbert.indexer \
			-i datasets/${DATASET}/L.${LABEL_EMB}.npz \
			-o ${OUTPUT_DIR}/indexer \
			-d ${DEPTH} --algo ${ALGO} --seed ${SEED} --max-iter 20

		# ranker (default: matcher=hierarchical linear)
    RANKER_DIR=${OUTPUT_DIR}/ranker
		mkdir -p ${OUTPUT_DIR}/ranker
		python -m xbert.ranker train \
			-x datasets/${DATASET}/X.trn.npz \
			-y datasets/${DATASET}/Y.trn.npz \
			-c ${OUTPUT_DIR}/indexer/code.npz \
			-o ${RANKER_DIR} -t 0.01

    PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.linear.npz
		python -m xbert.ranker predict \
      -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
			-x datasets/${DATASET}/X.tst.npz \
			-y datasets/${DATASET}/Y.tst.npz
    # append
    PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
	done
done

# final eval
EVAL_DIR=results_linear/${DATASET}
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator -y datasets/${DATASET}/Y.tst.npz -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${EXP_NAME}.txt

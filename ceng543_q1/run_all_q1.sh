#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
PYTHON=python

# activate conda env if you want (uncomment and change name)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ceng543

mkdir -p models outputs logs

# experiment function
run_exp () {
  MODE=$1   # glove|bert
  RNN=$2    # lstm|gru
  OUTDIR=models/${MODE}_${RNN}
  mkdir -p ${OUTDIR}
  echo "=== Running ${MODE} + ${RNN} ===" | tee logs/run_all.log
  if [ "${MODE}" = "glove" ]; then
    BATCH=64
    $PYTHON src/train.py --mode glove --rnn_type ${RNN} --epochs 5 --batch_size ${BATCH} --hidden_dim 128 --out_dir ${OUTDIR} --num_layers 2 --dropout 0.2 --lr 1e-3 --clip 1.0 2>&1 | tee ${OUTDIR}/train_stdout.log
  else
    BATCH=16
    $PYTHON src/train.py --mode bert --rnn_type ${RNN} --epochs 3 --batch_size ${BATCH} --hidden_dim 128 --out_dir ${OUTDIR} --num_layers 2 --freeze_bert --dropout 0.2 --lr 1e-4 --clip 1.0 2>&1 | tee ${OUTDIR}/train_stdout.log
  fi

  BEST=model_path="${OUTDIR}/best_${MODE}_${RNN}.pt"
  if [ ! -f "${OUTDIR}/best_${MODE}_${RNN}.pt" ]; then
    echo "Warning: best model not found at ${BEST}, trying last checkpoint..." | tee -a ${OUTDIR}/train_stdout.log
    # fallback: try to find any .pt
    BEST=$(ls ${OUTDIR}/*.pt 2>/dev/null | head -n1 || true)
    if [ -z "${BEST}" ]; then
      echo "No model file found for ${MODE}_${RNN}, skipping extract/visualize." | tee -a ${OUTDIR}/train_stdout.log
      return
    fi
  fi

  # extract embeddings
  echo "Extracting embeddings for ${MODE}_${RNN} from ${BEST}" | tee -a ${OUTDIR}/train_stdout.log
  $PYTHON src/extract_embeddings.py --mode ${MODE} --rnn_type ${RNN} --model_path ${BEST} --out_dir outputs --out_name ${MODE}_${RNN}_emb --batch_size 64 --hidden_dim 128

  # visualize
  echo "Visualizing ${MODE}_${RNN}" | tee -a ${OUTDIR}/train_stdout.log
  $PYTHON src/visualize_q1.py --emb_file outputs/${MODE}_${RNN}_emb.npz --out_dir outputs --prefix ${MODE}_${RNN}

  echo "=== Done ${MODE} + ${RNN} ===" | tee -a ${OUTDIR}/train_stdout.log
}

# run the four experiments sequentially
run_exp glove lstm
run_exp glove gru
run_exp bert lstm
run_exp bert gru

echo "All experiments finished. Outputs in models/ and outputs/"

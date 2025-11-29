# Question 1: Sentiment Classification with GloVe and BERT Embeddings

This question implements sentiment classification models using GloVe and BERT embeddings with bidirectional LSTM and GRU architectures on the IMDB dataset.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare GloVe embeddings:
   - Download GloVe 6B 300d embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
   - Convert to word2vec format and save to `~/.vector_cache/glove.6B.300d.word2vec.txt`
   - The script expects this file to exist for GloVe experiments

3. Download spaCy English model (if needed):
```bash
python -m spacy download en_core_web_sm
```

## Running Experiments

To reproduce all results, simply run:

```bash
bash run_all_q1.sh
```

This script will:
- Create necessary directories (`models/`, `outputs/`, `logs/`)
- Run four experiments sequentially:
  - GloVe + LSTM
  - GloVe + GRU
  - BERT + LSTM
  - BERT + GRU
- For each experiment, it will:
  1. Train the model and save checkpoints to `models/{mode}_{rnn}/`
  2. Extract embeddings from the best model
  3. Generate visualization plots (PCA and t-SNE) in `outputs/`

## Output Structure

After running the script, you will find:
- `models/`: Trained model checkpoints for each configuration
- `outputs/`: Extracted embeddings (`.npz` files) and visualization plots
- `logs/`: Training logs and stdout outputs

## Manual Execution

If you need to run individual components:

1. **Training**:
```bash
python src/train.py --mode glove --rnn_type lstm --epochs 5 --batch_size 64 --hidden_dim 128 --out_dir models/glove_lstm --num_layers 2 --dropout 0.2 --lr 1e-3 --clip 1.0
```

2. **Extract embeddings**:
```bash
python src/extract_embeddings.py --mode glove --rnn_type lstm --model_path models/glove_lstm/best_glove_lstm.pt --out_dir outputs --out_name glove_lstm_emb --batch_size 64 --hidden_dim 128
```

3. **Visualize embeddings**:
```bash
python src/visualize_embeddings.py --emb_npz outputs/glove_lstm_emb.npz --out_dir outputs --prefix glove_lstm
```

## Notes

- The script uses Python from your PATH. To use a conda environment, uncomment and modify the conda activation lines in `run_all_q1.sh`
- BERT experiments use DistilBERT and have frozen BERT layers by default
- Training logs are saved to both `logs/run_all.log` and individual experiment directories


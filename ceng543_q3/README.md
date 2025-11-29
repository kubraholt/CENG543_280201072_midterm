# Question 3: Seq2Seq vs Transformer Architectures with Embedding Paradigms

This question compares recurrent encoder-decoder (Seq2Seq) and Transformer architectures for neural machine translation, evaluating different embedding paradigms (learnable, GloVe, DistilBERT) on the Multi30K dataset (English to German).

## Setup

1. Install dependencies:
```bash
pip install -r requirements_q3.txt
```

2. Prepare GloVe embeddings (if using GloVe mode):
   - Download GloVe 6B 300d embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
   - Convert to word2vec format and save to `~/.vector_cache/glove.6B.300d.word2vec.txt`

3. The Multi30K dataset will be automatically downloaded from HuggingFace datasets when you run the training script.

## Running Experiments

To reproduce all results, simply run:

```bash
bash run_all_q3_experiments.sh
```

This script will run the following experiments sequentially:

### Q3.a: Seq2Seq + Bahdanau Attention
- Seq2Seq with learnable embeddings
- Seq2Seq with GloVe embeddings
- Seq2Seq with DistilBERT embeddings

### Q3.b: Transformer Architecture
- Transformer with learnable embeddings (3 layers, 8 heads)
- Transformer with GloVe embeddings (3 layers, 8 heads)
- Transformer with DistilBERT embeddings (3 layers, 8 heads)

### Q3.e: Ablation Study
- Transformer with 2 layers, 4 heads
- Transformer with 4 layers, 8 heads
- Transformer with 6 layers, 8 heads

After all experiments complete, the script automatically collects results and generates a summary CSV file (`q3_results/summary.csv`) with BLEU and ROUGE-L scores for all configurations.

## Output Structure

After running the script, you will find:
- `q3_experiments/`: Individual experiment directories, each containing:
  - `checkpoints/`: Model checkpoints
  - `outputs/`: Generated translations and predictions
  - `logs/`: Training metrics (CSV), test results (JSON), and configuration (JSON)
- `q3_results/`: 
  - `summary.csv`: Summary of all experiments with BLEU and ROUGE-L scores
  - `log_*.txt`: Individual experiment log files

## Additional Analysis

After running the main experiments, you can analyze results in detail:

```bash
python analyze_q3_results.py
```

This script generates:
- Comparison plots between Seq2Seq and Transformer
- Embedding paradigm comparisons
- Ablation study visualizations
- Performance metrics tables

## Manual Execution

If you need to run individual experiments:

1. **Seq2Seq with learnable embeddings**:
```bash
python train_q3_complete.py --model seq2seq --emb_mode learnable --epochs 12 --batch 64 --lr 1e-3 --exp_name seq2seq_learnable
```

2. **Transformer with GloVe embeddings**:
```bash
python train_q3_complete.py --model transformer --emb_mode glove --d_model 256 --n_layers 3 --n_heads 8 --d_ff 512 --epochs 12 --batch 64 --lr 1e-3 --exp_name transformer_glove
```

3. **Ablation: Transformer with custom layers/heads**:
```bash
python train_q3_complete.py --model transformer --emb_mode learnable --n_layers 4 --n_heads 4 --d_model 256 --d_ff 512 --epochs 12 --batch 64 --exp_name transformer_L4H4
```

## Testing Setup

To verify your setup works correctly before running full experiments:

```bash
python test_q3_setup.py
```

This runs quick sanity checks with minimal epochs (2 epochs) for both Seq2Seq and Transformer models.

## Notes

- Default hyperparameters: embedding dimension 256, hidden dimensions 256, vocabulary size 10000
- DistilBERT experiments use smaller batch sizes (32) due to higher memory requirements
- Training metrics (loss, BLEU, ROUGE-L) are logged per epoch in CSV format
- Test set evaluation is performed automatically after training completes
- All experiments use the same random seed (42) for reproducibility
- The script automatically generates experiment names if not specified


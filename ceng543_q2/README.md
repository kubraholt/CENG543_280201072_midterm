# Question 2: Machine Translation with Different Attention Mechanisms

This question implements neural machine translation models using three different attention mechanisms (Bahdanau, Luong, and Scaled Dot Product) on the Multi30K dataset (English to German translation).

## Setup

1. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets sacrebleu rouge-score
pip install numpy matplotlib tqdm
```

2. The dataset (Multi30K) will be automatically downloaded from HuggingFace datasets when you run the training script.

## Running Experiments

To reproduce all results, simply run:

```bash
bash run_all.sh
```

This script will:
- Train three models sequentially, one for each attention mechanism:
  - Bahdanau attention
  - Luong attention
  - Scaled Dot Product attention
- For each attention type, it will:
  1. Modify the training script to use the specified attention mechanism
  2. Train the model and save checkpoints to `checkpoints/{attention_type}/`
  3. Save training metrics to `logs/{attention_type}/metrics.csv`
  4. Evaluate the best model using BLEU, ROUGE-L, and perplexity metrics
  5. Save evaluation results to `run_logs/eval_{attention_type}.log`

## Output Structure

After running the script, you will find:
- `checkpoints/`: Model checkpoints organized by attention type (`bahdanau/`, `luong/`, `scaleddot/`)
- `outputs/`: Generated translations and attention weights for each attention type
- `logs/`: Training metrics (CSV files) for each attention type
- `run_logs/`: Training and evaluation log files

## Additional Analysis Scripts

After running the main experiments, you can use these scripts for further analysis:

1. **Collect results summary**:
```bash
python collect_results.py
```
This creates `results_summary.csv` with BLEU, ROUGE-L, and perplexity scores for all attention types.

2. **Compute attention statistics**:
```bash
python compute_attn_stats.py
```
This analyzes attention weights and creates `attn_stats_summary.csv` with entropy and sharpness metrics for each attention mechanism.

## Manual Execution

If you need to run individual components:

1. **Training a single model**:
```bash
python train_multi30k_attention.py --attn bahdanau --epochs 12 --batch 64
```

2. **Evaluation**:
```bash
python eval_metrics.py --ckpt checkpoints/bahdanau/best.pt --attn bahdanau --batch 64
```

## Notes

- The script temporarily modifies `train_multi30k_attention.py` to change the attention mechanism, then restores the original file
- Default hyperparameters: embedding dimension 256, encoder/decoder hidden size 256, vocabulary size 10000, teacher forcing ratio 0.5
- Training logs are saved to both individual attention directories and the main `run_logs/` directory
- The evaluation script computes BLEU, ROUGE-L, and perplexity on the test set


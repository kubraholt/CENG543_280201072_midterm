# Question 5: Model Interpretability and Error Analysis

This question performs comprehensive interpretability analysis and error analysis on the best-performing model from Q3 (Transformer + DistilBERT) for neural machine translation.

## Prerequisites

Before running Q5, ensure that Q3 experiments have been completed, as Q5 loads the best model checkpoint from Q3:
- Model: Transformer + DistilBERT (3 layers, 8 heads)
- Expected location: `../ceng543_q3/q3_experiments/transformer_distilbert_L3H8/checkpoints/best.pt`

## Setup

1. Install dependencies:
```bash
pip install -r requirements_q5.txt
```

2. Ensure Q3 experiments are complete (see Q3 README for details).

## Running the Complete Analysis Pipeline

To reproduce all interpretability and error analysis results, simply run:

```bash
bash run_all_q5.sh
```

This script executes the following steps sequentially:

1. **Load Best Model**: Loads the Transformer + DistilBERT checkpoint from Q3
2. **Attention Visualization**: Generates attention heatmaps for example translations
3. **Integrated Gradients**: Computes input token attributions using gradient-based methods
4. **LIME Analysis**: Generates local interpretable explanations for model predictions
5. **Failure Case Analysis**: Identifies and categorizes 5 representative translation failures
6. **Uncertainty Quantification**: Measures model confidence via output entropy and calibration
7. **Visualize Results**: Creates summary visualizations and dashboard

## Output Structure

After running the pipeline, you will find:

- `outputs/`:
  - `model_info.json`: Information about the loaded model
  - `attention_heatmaps/`: Attention visualization plots for example sentences
  - `integrated_gradients/`: Attribution visualizations showing important input tokens
  - `lime_explanations/`: LIME-based explanations and feature importance plots
  - `failure_cases.json`: Categorized failure cases with analysis
  - `uncertainty_metrics.json`: Entropy and calibration metrics
  - `summary/`: Summary dashboard and comparison plots

## Analysis Methods

### Attention Visualization
- Visualizes multi-head attention patterns across encoder-decoder layers
- Shows which source tokens the model attends to when generating target tokens
- Identifies alignment patterns and potential issues

### Integrated Gradients
- Gradient-based attribution method for input tokens
- Quantifies the contribution of each input token to the output
- Helps identify which parts of the source sentence drive the translation

### LIME Analysis
- Local Interpretable Model-agnostic Explanations
- Generates interpretable explanations by perturbing inputs
- Compares feature importance across different examples

### Failure Case Analysis
Categorizes failures into:
- Rare Word (OOV): Out-of-vocabulary words
- Long-Distance Dependency: Complex syntactic structures
- Negation Handling: Logical operators and contrastive structures
- Ambiguous Pronoun Reference: Coreference resolution issues
- Idiomatic Expressions: Culture-specific phrases

### Uncertainty Quantification
- Computes output entropy to measure prediction confidence
- Analyzes calibration: correlation between confidence and accuracy
- Identifies cases where the model is overconfident or underconfident

## Manual Execution

If you need to run individual components:

1. **Load model**:
```bash
python 1_load_best_model.py
```

2. **Attention visualization**:
```bash
python 2_attention_visualization.py
```

3. **Integrated Gradients**:
```bash
python 3_integrated_gradients.py
```

4. **LIME analysis**:
```bash
python 4_lime_analysis.py
```

5. **Failure case analysis**:
```bash
python 5_failure_case_analysis.py
```

6. **Uncertainty quantification**:
```bash
python 6_uncertainty_quantification.py
```

7. **Visualize results**:
```bash
python 7_visualize_results.py
```

## Notes

- The analysis uses the Multi30K test set from Q3
- All visualizations are saved as PNG files in the respective output directories
- Failure cases are manually curated examples representing common error patterns
- Uncertainty analysis requires model predictions on the test set
- Integrated Gradients and LIME may take longer to compute for large examples
- The model architecture is defined in `model_with_attention.py` with attention extraction capabilities


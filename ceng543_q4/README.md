# Question 4: Retrieval-Augmented Generation (RAG) System

This question implements a complete RAG pipeline comparing sparse (BM25) and dense (Sentence-BERT) retrieval methods with FLAN-T5-base as the generator on the HotpotQA dataset.

## Setup

1. Install dependencies:
```bash
pip install -r requirements_q4.txt
```

2. The HotpotQA dataset will be automatically downloaded from HuggingFace datasets when you run the pipeline.

## Running the Complete Pipeline

To reproduce all results, simply run:

```bash
bash run_all_q4.sh
```

This script executes the following steps sequentially:

1. **Data Preparation**: Downloads and preprocesses HotpotQA validation set
2. **Build BM25 Retriever**: Creates sparse lexical retrieval index
3. **Build Sentence-BERT Retriever**: Creates dense semantic retrieval index
4. **Evaluate Retrieval**: Compares BM25 and SBERT retrieval performance
5. **Oracle Generation**: Generates answers using gold passages (upper bound)
6. **BM25 RAG Pipeline**: End-to-end RAG with BM25 retrieval + FLAN-T5-base
7. **SBERT RAG Pipeline**: End-to-end RAG with Sentence-BERT retrieval + FLAN-T5-base
8. **Evaluate Generation**: Computes BLEU, ROUGE, and BERTScore metrics
9. **Qualitative Analysis**: Generates example outputs for manual inspection
10. **Visualize Results**: Creates plots comparing retrieval and generation performance

## Output Structure

After running the pipeline, you will find:

- `data/`:
  - `corpus.json`: Preprocessed passage corpus
  - `questions.json`: Question-answer pairs with ground truth

- `models/`:
  - `bm25_index.pkl`: BM25 retrieval index
  - `sbert_embeddings.npy`: Sentence-BERT passage embeddings
  - `sbert_passage_ids.json`: Passage ID mapping for SBERT

- `outputs/`:
  - `retrieval_metrics.json`: Retrieval evaluation results (recall@k, MRR)
  - `generation_metrics.json`: Generation evaluation results (BLEU, ROUGE, BERTScore)
  - `qualitative_examples.md`: Example outputs for qualitative analysis
  - `bm25_rag_predictions.json`: BM25 RAG predictions
  - `sbert_rag_predictions.json`: SBERT RAG predictions
  - `oracle_predictions.json`: Oracle (gold passage) predictions
  - `plots/`: Visualization plots comparing methods

## Manual Execution

If you need to run individual components:

1. **Prepare data**:
```bash
python 1_prepare_data.py
```

2. **Build retrievers**:
```bash
python 2_build_bm25_retriever.py
python 3_build_sbert_retriever.py
```

3. **Evaluate retrieval**:
```bash
python 4_evaluate_retrieval.py
```

4. **Run RAG pipelines**:
```bash
python 6_rag_bm25.py
python 7_rag_sbert.py
```

5. **Evaluate generation**:
```bash
python 8_evaluate_generation.py
```

6. **Generate visualizations**:
```bash
python 10_visualize_results.py
```

## System Components

- **Retrieval Methods**:
  - BM25: Sparse lexical retrieval using term frequency
  - Sentence-BERT: Dense semantic retrieval using `all-MiniLM-L6-v2` embeddings

- **Generator**: FLAN-T5-base (instruction-tuned T5 model)

- **Evaluation Metrics**:
  - Retrieval: Recall@k, Mean Reciprocal Rank (MRR)
  - Generation: BLEU, ROUGE-L, BERTScore

## Notes

- The pipeline uses the HotpotQA validation set (distractor setting)
- Default retrieval: top-k=5 passages per question
- Sentence-BERT model automatically uses GPU if available
- FLAN-T5-base is loaded from HuggingFace and uses GPU if available
- All intermediate results are saved to allow partial re-runs
- The oracle generation (step 5) provides an upper bound on generation quality
- Qualitative analysis (step 9) generates markdown examples for manual review


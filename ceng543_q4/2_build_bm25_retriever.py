"""
Q4 - Build BM25 Retriever
Indexes corpus using BM25 for sparse lexical retrieval.
"""

import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def build_bm25_index():
    print("="*60)
    print("Q4: Building BM25 Retriever")
    print("="*60)
    
    print("\n[1/2] Loading corpus...")
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    print(f"Loaded {len(corpus)} passages")
    
    print("\n[2/2] Building BM25 index...")
    tokenized_corpus = []
    passage_ids = []
    
    for passage in tqdm(corpus, desc="Tokenizing"):
        tokens = passage['text'].lower().split()
        tokenized_corpus.append(tokens)
        passage_ids.append(passage['id'])
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("\nSaving BM25 index...")
    with open("models/bm25_index.pkl", "wb") as f:
        pickle.dump({
            'bm25': bm25,
            'passage_ids': passage_ids
        }, f)
    
    print("\n" + "="*60)
    print("BM25 index built successfully!")
    print(f"Saved to: models/bm25_index.pkl")
    print(f"Indexed {len(passage_ids)} passages")
    print("="*60)

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    build_bm25_index()
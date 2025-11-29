"""
Q4 - Data Preparation: HotpotQA Dataset
Downloads and preprocesses HotpotQA dev set for RAG evaluation.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

def prepare_hotpotqa():
    print("="*60)
    print("Q4 Data Preparation: HotpotQA Dataset")
    print("="*60)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("\n[1/3] Loading HotpotQA dev set from HuggingFace...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"Loaded {len(dataset)} questions")
    
    print("\n[2/3] Building corpus from context paragraphs...")
    corpus = {}
    questions_data = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        question = example['question']
        answer = example['answer']
        context = example['context']
        supporting_facts = example['supporting_facts']
        
        # Extract unique paragraphs into corpus
        paragraph_ids = []
        
        # HotpotQA context structure: {'title': [...], 'sentences': [...]}
        titles = context['title']
        sentences_list = context['sentences']
        
        for title, sentences in zip(titles, sentences_list):
            passage_text = " ".join(sentences)
            passage_id = f"{title}_{hash(passage_text) % 1000000}"
            
            if passage_id not in corpus:
                corpus[passage_id] = {
                    'id': passage_id,
                    'title': title,
                    'text': passage_text
                }
            
            paragraph_ids.append(passage_id)
        
        # Build gold supporting facts mapping
        gold_passage_ids = []
        for sf_title, sf_sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
            for pid, pdata in corpus.items():
                if pdata['title'] == sf_title:
                    gold_passage_ids.append(pid)
                    break
        
        questions_data.append({
            'id': f"q_{idx}",
            'question': question,
            'answer': answer,
            'context_ids': paragraph_ids,
            'gold_passage_ids': list(set(gold_passage_ids))
        })
    
    print(f"Built corpus with {len(corpus)} unique passages")
    print(f"Processed {len(questions_data)} questions")
    
    print("\n[3/3] Saving preprocessed data...")
    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(list(corpus.values()), f, indent=2, ensure_ascii=False)
    
    with open("data/questions.json", "w", encoding="utf-8") as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print(f"Corpus: data/corpus.json ({len(corpus)} passages)")
    print(f"Questions: data/questions.json ({len(questions_data)} items)")
    print("="*60)
    
    return corpus, questions_data

if __name__ == "__main__":
    prepare_hotpotqa()
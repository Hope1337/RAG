from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch
from tqdm import tqdm
from src.retrievers.custome import CustomeRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = load_dataset("ArchitRastogi/USLawQA", split="train", streaming=True)
texts_list = []
for sample in tqdm(dataset, desc="Loading dataset"):
    texts_list.append((sample["index"], sample["document"]))

corpus = {doc[0]:doc[1] for doc in texts_list}
#print(corpus)

dataset = load_dataset("ArchitRastogi/USLawQA", split="test", streaming=True)
test_data = []
for sample in dataset:
    for law_number, questions in sample.items():
        for question in questions:
            test_data.append((law_number, question))

queries      = [td[1] for td in test_data]
ground_truth = [td[0] for td in test_data]

def evaluate_retrieval(corpus, queries, ground_truth, k=10):
    retriever = CustomeRetriever()
    results = retriever.retrieve(corpus, queries, topk=k)

    ranks = []  
    for idx, law_number in enumerate(ground_truth):
        retrieved_law_numbers = [td['corpus_id'] for td in results[idx]] 
        if law_number in retrieved_law_numbers:
            rank = retrieved_law_numbers.index(law_number) + 1  
        else:
            rank = -1  

        ranks.append(rank)

    return ranks

retrieval_ranks = evaluate_retrieval(corpus, queries, ground_truth)

print("🔹 Tỷ lệ tìm thấy đúng trong top-k:", sum(r >= 1 for r in retrieval_ranks) / len(retrieval_ranks))
print("🔹 Trung bình vị trí tìm thấy:", sum(r for r in retrieval_ranks if r > 0) / sum(1 for r in retrieval_ranks if r > 0))

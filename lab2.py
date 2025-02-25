
from src.retrievers.hybrid import Aggregator, Ranker
import pandas as pd
from os.path import exists, join
import numpy as np
from ragatouille import RAGPretrainedModel

import bm25s
import Stemmer  # optional: for stemming
from sentence_transformers import SentenceTransformer
import faiss
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from torch.cuda import empty_cache
from src.retrievers.hybrid import Aggregator


def hugging_search(raw_corpus, queries, model_name, topk=5):
        model = SentenceTransformer(model_name)
        test = ['hi']
        dimension = model.encode(test).shape[1]
        index = faiss.IndexFlatL2(dimension) 

        idx2id = {i:pid for i, pid in enumerate(raw_corpus.keys())}
        corpus = list(raw_corpus.values())

        embeddings = model.encode(corpus, normalize_embeddings=True)
        index.add(embeddings)

        embeddings = model.encode(queries, normalize_embeddings=True)
        distances, indices = index.search(embeddings, topk)

        results = []
        for i in range(len(queries)):
            result_one_q = []
            for idx, score in zip(indices[i], distances[i]):
                result_one_q.append({'corpus_id': idx2id[idx], 'score': 1.0 - score})
            results.append(result_one_q)

        return results

def bm25_search(raw_corpus, queries, topk):
    stemmer = Stemmer.Stemmer("english")
    corpus  = raw_corpus.values()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(queries, stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=topk)

    idx2id = {idx:id for idx, id in enumerate(raw_corpus.keys())}

    final_results = []

    for j in range(results.shape[0]):
        result_one_q = []
        for i in range(results.shape[1]):
            doc, score = results[1, i], scores[1, i]
            result_one_q.append({'corpus_id': idx2id[doc], 'score': score})
        final_results.append(result_one_q)
    
    return final_results


def colbert_search(corpus, queries, model_name_or_path: str, output_dir: str = 'output', return_topk: int = None):
    documents = list(corpus.values())
    idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

    index_root = f"output/testing/indexes/{model_name_or_path.split('/')[-1]}"
    if not exists(join(index_root, "lleqa.index")):
        with Run().context(RunConfig(nranks=1, index_root=index_root)):
            indexer = Indexer(checkpoint=model_name_or_path, config=ColBERTConfig(query_maxlen=64, doc_maxlen=512, gpus=0))
            indexer.index(name="lleqa.index", collection=documents, overwrite='reuse')

    with Run().context(RunConfig(nranks=1, index_root=index_root)):
        searcher = Searcher(index="lleqa.index", config=ColBERTConfig(query_maxlen=64, doc_maxlen=512, gpus=0))
        ranked_lists = searcher.search_all({idx: query for idx, query in enumerate(queries)}, k=return_topk or len(documents))

    empty_cache()
    return [[{'corpus_id': idx2id.get(x[0]), 'score': x[2]} for x in res] for res in ranked_lists.todict().values()]

def colbert_rerank(raw_results, query, topk):
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    return RAG.rerank(query=query, documents=raw_results, k=topk)

class CustomeRetriever:
     
     def __init__(self):
        self.run_bm25 = True
        self.run_e5   = True
        self.run_baai = True

        self.topk_bm25 = 10
        self.topk_e5   = 10
        self.topk_baai = 10
        pass 
     
     def retrieve(self, corpus, queries, topk):
        results = {}

        if self.run_bm25:
            results['bm25'] = bm25_search(corpus, queries, self.topk_bm25)
        
        if self.run_e5:
            results['e5'] = hugging_search(corpus, queries, 'intfloat/e5-large-v2', self.topk_e5)
        
        if self.run_baai:
            results['baai'] = hugging_search(corpus, queries, 'BAAI/bge-large-zh-v1.5', self.topk_baai)
        
        ranked_list = Aggregator.fuse(ranked_lists=results, method='rrf')

        final_ranked_list = []

        for qid, t in enumerate(ranked_list):
            temp_corpus = []
            temp_idx2id = {}
            for tid, tt in enumerate(t):
                temp_corpus.append(corpus[tt['corpus_id']])
                temp_idx2id[tid] = tt['corpus_id']
            
            colbert_list = colbert_rerank(temp_corpus, queries[qid], topk=topk)
            #print(colbert_list)

            final_ranked_list.append([{'corpus_id': temp_idx2id[i['result_index']], 'score':i['score']} for i in colbert_list])
        
        return final_ranked_list

retriever = CustomeRetriever()

corpus = {
    1: "Machine learning is a field of artificial intelligence.",
    2: "Deep learning is a subset of machine learning focused on neural networks.",
    3: "Support vector machines are a type of supervised learning algorithm.",
    4: "Reinforcement learning involves training agents through rewards and penalties.",
    5: "Gradient descent is an optimization algorithm used in machine learning.",
    6: "Neural networks are the backbone of deep learning.",
    7: "Convolutional neural networks (CNNs) are used in image processing tasks.",
    8: "Recurrent neural networks (RNNs) are useful for sequential data such as text and speech.",
    9: "Transfer learning allows models to leverage pre-trained knowledge for new tasks.",
    10: "Unsupervised learning algorithms, such as clustering, do not rely on labeled data.",
    11: "Supervised learning requires labeled datasets to train predictive models.",
    12: "Semi-supervised learning combines small amounts of labeled data with large unlabeled datasets.",
    13: "Self-supervised learning is gaining popularity in natural language processing.",
    14: "Transformers are deep learning models designed for NLP tasks.",
    "hihi": "BERT and GPT are transformer-based models used for language understanding.",
    "haha": "Reinforcement learning can be applied to robotics and game playing.",
    17: "Gradient boosting is a powerful technique for structured data tasks.",
    18: "Random forests combine multiple decision trees to improve accuracy.",
    "test": "Principal component analysis (PCA) is used for dimensionality reduction.",
    20: "Feature engineering improves the performance of machine learning models.",
    21: "Hyperparameter tuning is essential for optimizing machine learning models.",
    22: "Bayesian optimization is a method for hyperparameter tuning.",
    23: "Overfitting occurs when a model learns noise instead of patterns.",
    24: "Regularization techniques such as L1 and L2 help reduce overfitting.",
    25: "Cross-validation helps assess the performance of machine learning models."
}
    
queries = ["What is BERT?", "What is PCA?"]

ranked_list = retriever.retrieve(corpus, queries, topk=5)
print(ranked_list)

for t in ranked_list:
    print(len(t))
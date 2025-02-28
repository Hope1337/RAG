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
import os
from sentence_transformers import CrossEncoder


def hugging_search(raw_corpus, queries, model_name, topk=5):
    index_path = './' + model_name.split('/')[1] + 'faiss.bin'
    model = SentenceTransformer(model_name)
    idx2id = {i:pid for i, pid in enumerate(raw_corpus.keys())}
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)
    else:
        test = ['hi']
        dimension = model.encode(test).shape[1]
        index = faiss.IndexFlatL2(dimension) 

        corpus = list(raw_corpus.values())

        embeddings = model.encode(corpus, normalize_embeddings=True)
        index.add(embeddings)
        faiss.write_index(index, index_path)

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
            doc, score = results[j, i], scores[j, i]
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

def rank_documents(corpus, query, topk):
    """
    Rank documents using the cross-encoder model.
    
    :param corpus: List of documents (strings)
    :param query: Query string
    :param topk: Number of top relevant documents to return
    :return: List of dictionaries with 'result_index' and 'score'
    """
    # Load the cross-encoder model
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    # Create input pairs (query, document)
    pairs = [(query, doc) for doc in corpus]
    
    # Compute relevance scores
    scores = cross_encoder_model.predict(pairs)
    
    # Rank documents by score
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    return [{"result_index": idx, "score": score} for idx, score in ranked_results[:topk]]

class CustomeRetriever:
     
     def __init__(self):
        self.run_bm25 = True
        self.run_e5   = True
        self.run_baai = True
        self.run_mixed_bread = True

        self.topk_bm25 = 10
        self.topk_e5   = 10
        self.topk_baai = 10
        self.topk_mixed_bread = 10
        pass 
     
     def retrieve(self, corpus, queries, topk):
        results = {}

        if self.run_mixed_bread:
            print('Running mixed bread...')
            results['mixed_bread'] = hugging_search(corpus, queries, 'mixedbread-ai/mxbai-embed-large-v1', self.topk_mixed_bread)
            print('Completed!')
            print()

        if self.run_bm25:
            print('Running BM25...')
            results['bm25'] = bm25_search(corpus, queries, self.topk_bm25)
            print('Completed!')
            print()
        
        if self.run_e5:
            print('Running e5')
            results['e5'] = hugging_search(corpus, queries, 'intfloat/e5-large-v2', self.topk_e5)
            print('Completed!')
            print()
        
        if self.run_baai:
            print('Running baai')
            results['baai'] = hugging_search(corpus, queries, 'BAAI/bge-large-en-v1.5', self.topk_baai)
            print('Completed!')
            print()

        print('Fusing') 
        ranked_list = Aggregator.fuse(ranked_lists=results, method='rrf')
        print('Completed!')
        print()

        print('Reranking')
        final_ranked_list = []

        for qid, t in enumerate(ranked_list):
            temp_corpus = []
            temp_idx2id = {}
            for tid, tt in enumerate(t):
                temp_corpus.append(corpus[tt['corpus_id']])
                temp_idx2id[tid] = tt['corpus_id']
            
            #colbert_list = colbert_rerank(temp_corpus, queries[qid], topk=topk)
            colbert_list = rank_documents(temp_corpus, queries[qid], topk=topk)
            #print(colbert_list)

            final_ranked_list.append([{'corpus_id': temp_idx2id[i['result_index']], 'score':i['score']} for i in colbert_list])
        print('Completed!')
        print()
        
        return final_ranked_list
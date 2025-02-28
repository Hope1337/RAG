import os
import sys
import copy
import pathlib
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from os.path import exists, join
from collections import defaultdict, OrderedDict

import math
import torch
import random
import itertools
import numpy as np
import pandas as pd
from torch.cuda import empty_cache
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()
sys.path.append(str(pathlib.Path().resolve()))


def run_evaluation(predictions, labels, print2console: bool = True, log2wandb: bool = False, args: argparse.Namespace = None):
    from src.utils.metrics import Metrics
    from src.utils.loggers import WandbLogger

    evaluator = Metrics(recall_at_k=[5,10,20,50,100,200,500,1000], map_at_k=[10,100], mrr_at_k=[10,100], ndcg_at_k=[10,100])
    scores = evaluator.compute_all_metrics(all_ground_truths=labels, all_results=predictions)
    if print2console:
        for metric, score in scores.items():
            print(f'- {metric.capitalize()}: {score:.3f}')
    if log2wandb:
        logger = WandbLogger(
            project_name="lleqa", 
            run_name=f"hybrid-{args.eval_type}-{args.fusion}-{args.normalization+'-' if args.fusion == 'nsf' else ''}{'-'.join([n.split('_')[-1] for n, val in vars(args).items() if n.startswith('run_') and val is True])}",
            run_config=args, 
            log_dir=join(args.output_dir, 'logs'),
        )
        for metric, score in scores.items():
            logger.log_eval(0, 0, f'{args.data_split}/{metric}', score)
    return scores


class Ranker:
    """
    A class for ranking queries against a corpus.
    """
    @staticmethod
    def bm25_search(queries, corpus, do_preprocessing: bool, k1: float, b: float, return_topk: int = None):
        """
        Perform retrieval on a corpus using the BM25 algorithm.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param do_preprocessing: Whether to preprocess the articles before searching.
        :param k1: The k1 parameter for the BM25 algorithm.
        :param b: The b parameter for the BM25 algorithm.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from src.retrievers.bm25 import BM25
        from src.data.preprocessor import TextPreprocessor

        documents = list(corpus.values())
        idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

        if do_preprocessing:
            cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
            documents = cleaner.preprocess(documents, lemmatize=True)
            queries = cleaner.preprocess(queries, lemmatize=True)
        
        retriever = BM25(corpus=documents, k1=k1, b=b)
        ranked_lists = retriever.search_all(queries, top_k=return_topk or len(documents))
        return [[{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in res] for res in ranked_lists]

    @staticmethod
    def single_vector_search(queries, corpus, model_name_or_path: str, return_topk: int = None):
        """
        Perform retrieval on a corpus using a single vector search model.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from sentence_transformers import util
        from .splade.splade import SPLADE
        from src.utils.sentence_transformers import SentenceTransformerCustom
        
        documents = list(corpus.values())
        idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

        if 'splade' in model_name_or_path.lower():
            model = SPLADE(model_name_or_path, max_query_length=64, max_doc_length=512)
        else:
            model = SentenceTransformerCustom(model_name_or_path)
            model.max_seq_length = 512
        
        d_embs = model.encode(sentences=documents, batch_size=64, convert_to_tensor=True, show_progress_bar=True, **({'query_mode': False} if isinstance(model, SPLADE) else {}))
        q_embs = model.encode(sentences=queries, batch_size=64, convert_to_tensor=True, show_progress_bar=True, **({'query_mode': True} if isinstance(model, SPLADE) else {}))
        ranked_lists = util.semantic_search(query_embeddings=q_embs, corpus_embeddings=d_embs, top_k=return_topk or len(documents), score_function=util.cos_sim)

        del model, d_embs, q_embs; empty_cache()
        return [[{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in res] for res in ranked_lists]

    @staticmethod
    def multi_vector_search(queries, corpus, model_name_or_path: str, output_dir: str = 'output', return_topk: int = None):
        """
        Perform retrieval on a corpus using a multi-vector search model.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param output_dir: The output directory to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from colbert import Indexer, Searcher
        from colbert.infra import Run, RunConfig, ColBERTConfig

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

    @staticmethod
    def cross_encoder_search(queries, candidates, model_name_or_path: str, return_topk: int = None):
        """
        Perform retrieval on a corpus using a cross-encoder model.

        :param queries: A list of queries to rank.
        :param candidates: A list of dictionaries of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from src.utils.sentence_transformers import CrossEncoderCustom

        model = CrossEncoderCustom(model_name_or_path)

        ranked_lists = []
        for query, cands in zip(queries, candidates):
            documents = cands.values()
            idx2id = {i: pid for i, pid in enumerate(cands.keys())}

            results = model.rank(query, documents=documents, top_k=return_topk or len(documents), batch_size=64, show_progress_bar=True)
            ranked_lists.append([{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in results])
        
        del model; empty_cache()
        return ranked_lists
    
    @staticmethod
    def hugging_search(queries, raw_corpus, model_name, topk=5):
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

class Aggregator:
    """
    A class for aggregating ranked lists.
    """
    @classmethod
    def fuse(
        cls, 
        ranked_lists,
        method: str,
        normalization: str = None, 
        linear_weights = None, 
        percentile_distributions = None,
        return_topk: int = 1000,
    ):
        """
        Fuse the ranked lists of different retrieval systems.

        :param ranked_lists: A dictionary of ranked lists, where each key is a retrieval system and each value is a list of ranked lists.
        :param method: The fusion method to use.
        :param normalization: The normalization method to use.
        :param linear_weights: A dictionary of weights to use for linear combination.
        :param percentile_distributions: A dictionary of percentile distributions to use for percentile rank normalization.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the final ranked lists.
        """
        num_queries = len(next(iter(ranked_lists.values())))
        assert all(len(system_res) == num_queries for system_res in ranked_lists.values()), (
            "Ranked results from different retrieval systems have varying lenghts across systems (i.e., some systems have been run on more queries)."
        )
        assert (set(ranked_lists.keys()) == set(linear_weights.keys()), (
            "The system names in the provided 'linear_weights' dictionary for convex combination do not correspond to those from the 'ranked_lists'."
        )) if method == 'nsf' else True

        final_results = []
        for i in range(num_queries):
            query_results = []

            for system, results in ranked_lists.items():
                res = cls.convert2dict(results[i])

                if method == 'bcf':
                    res = cls.transform_scores(res, transformation='borda-count')
                
                elif method == 'rrf':
                    res = cls.transform_scores(res, transformation='reciprocal-rank')
                
                elif method == 'nsf':
                    res = cls.transform_scores(res, transformation=normalization, percentile_distr=percentile_distributions.get(system))
                    res = cls.weight_scores(res, w=linear_weights[system])
                
                query_results.append(res)
            
            final_results.append(cls.aggregate_scores(*query_results)[:return_topk])
        
        return final_results

    @staticmethod
    def convert2dict(results):
        """
        Convert a list of "result_id: score" dictionaries to a dictionary of "result_id: score" items.

        :param results: A list of "result_id: score" dictionaries.
        :return: A dictionary of "result_id: score" items.
        """
        if sys.version_info >= (3, 7):
            return {res['corpus_id']: res['score'] for res in results}
        else:
            return OrderedDict((res['corpus_id'], res['score']) for res in results)

    @staticmethod
    def transform_scores(results, transformation: str, percentile_distr: np.array = None):
        """
        Transform the scores of a list of results.

        :param results: A dictionary of "result_id: score" items.
        :param transformation: The transformation method to use.
        :param percentile_distr: The percentile distribution to use for percentile rank normalization.
        :return: A dictionary of "result_id: score" items.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if transformation == 'borda-count':
            num_candidates = len(results)
            return {pid: (num_candidates - idx+1) / num_candidates for idx, pid in enumerate(results.keys())}
        
        elif transformation == 'reciprocal-rank':
            return {pid: 1 / (5 + idx) for idx, pid in enumerate(results.keys())}

        elif transformation == 'min-max':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            min_val, max_val = torch.min(scores), torch.max(scores)
            scores = (scores - min_val) / (max_val - min_val) if min_val != max_val else torch.ones_like(scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}

        elif transformation == 'z-score':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            mean_val, std_val = torch.mean(scores), torch.std(scores)
            scores = (scores - mean_val) / std_val if std_val != 0 else torch.zeros_like(scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
        
        elif transformation == 'arctan':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            scores = (2 / math.pi) * torch.atan(0.1 * scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
    
        elif transformation == 'percentile-rank' or transformation == 'normal-curve-equivalent':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            distribution = torch.tensor(percentile_distr, device=device, dtype=torch.float32)
            differences = torch.abs(distribution[:, None] - scores)
            scores = torch.argmin(differences, axis=0) / distribution.size(0)
            if transformation == 'normal-curve-equivalent':
                scores = torch.distributions.Normal(0, 1).icdf(scores / 100) * 21.06 + 50
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
    
        return results

    @staticmethod
    def weight_scores(results, w: float):
        """
        Weight the scores of a list of results.

        :param results: A dictionary of "result_id: score" items.
        :param w: The weight to apply.
        :return: A dictionary of "result_id: weighted_score" items.
        """
        return {corpus_id: score * w for corpus_id, score in results.items()}

    @staticmethod
    def aggregate_scores(*args):
        """
        Aggregate the scores of a list of lists of results.

        :param args: A list of dictionaries of "result_id: score" items.
        :return: A list of dictionaries of "result_id: score" items.
        """
        agg_results = defaultdict(float)
        for results in args:
            for pid, score in results.items():
                agg_results[pid] += score
        
        agg_results = sorted(agg_results.items(), key=lambda x: x[1], reverse=True)
        return [{'corpus_id': pid, 'score': score} for pid, score in agg_results]


def main(args):
    sep = f"#{'-'*40}#"
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    model_ckpts = {
        'dpr': {
            'general': 'antoinelouis/biencoder-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/dpr-legal-french'
        },
        'splade': {
            'general': 'antoinelouis/spladev2-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/splade-legal-french'
        },
        'colbert': {
            'general': 'antoinelouis/colbertv1-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/colbert-legal-french'
        },
        'monobert': {
            'general': 'antoinelouis/colbertv1-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/monobert-legal-french'
        }
    }
    args.eval_type = ('in' if args.models_domain == 'legal' else 'out') + 'domain'

    print("Loading corpus and queries...")
    #dfC = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
    #corpus = dfC.set_index('id')['article'].to_dict()
    #dfQ = load_dataset('maastrichtlawtech/lleqa', name='questions', split='validation' if args.data_split == 'dev' else args.data_split, token=os.getenv('HF')).to_pandas()
    #qids, queries, pos_pids = dfQ['id'].tolist(), dfQ['question'].tolist(), dfQ['article_ids'].tolist()

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
    15: "BERT and GPT are transformer-based models used for language understanding.",
    16: "Reinforcement learning can be applied to robotics and game playing.",
    17: "Gradient boosting is a powerful technique for structured data tasks.",
    18: "Random forests combine multiple decision trees to improve accuracy.",
    19: "Principal component analysis (PCA) is used for dimensionality reduction.",
    20: "Feature engineering improves the performance of machine learning models.",
    21: "Hyperparameter tuning is essential for optimizing machine learning models.",
    22: "Bayesian optimization is a method for hyperparameter tuning.",
    23: "Overfitting occurs when a model learns noise instead of patterns.",
    24: "Regularization techniques such as L1 and L2 help reduce overfitting.",
    25: "Cross-validation helps assess the performance of machine learning models."
}
    
    # Define a simple set of queries
    queries = [
        "What is deep learning?",
        "Explain reinforcement learning",
        "How does gradient descent work?"
    ]


    #------------------------------------------------------#
    #                       RETRIEVAL
    #------------------------------------------------------#
    if args.run_bm25:
        print(f"{sep}\n# Ranking with BM25\n{sep}")
        results['bm25'] = Ranker.bm25_search(queries, corpus, do_preprocessing=True, k1=2.5, b=0.2)

    if args.run_dpr:
        print(f"{sep}\n# Ranking with DPR\n{sep}")
        results['dpr'] = Ranker.single_vector_search(queries, corpus, model_name_or_path=model_ckpts['dpr'][args.models_domain])

    if args.run_splade:
        print(f"{sep}\n# Ranking with SPLADE\n{sep}")
        results['splade'] = Ranker.single_vector_search(queries, corpus, model_name_or_path=model_ckpts['splade'][args.models_domain])

    if args.run_colbert:
        print(f"{sep}\n# Ranking with ColBERT\n{sep}")
        results['colbert'] = Ranker.multi_vector_search(queries, corpus, model_name_or_path=model_ckpts['colbert'][args.models_domain])


    #------------------------------------------------------#
    #                       FUSION
    #------------------------------------------------------#
    distributions, weights = {}, {}
    if args.fusion == 'nsf':

        weights = {system: 1/len(results) for system in results}

        if args.normalization == 'percentile-rank' or args.normalization == 'normal-curve-equivalent':
            distr_df = pd.read_csv(join(args.output_dir, f"score_distributions_raw_{args.eval_type}_28k.csv"))
            distributions = {k: np.array(v) for k, v in distr_df.to_dict('series').items()}

    print(f"{sep}\n# Fusing results with {args.fusion.upper()}{' ('+args.normalization+')' if args.fusion == 'nsf' else ''}\n{sep}")
    ranked_lists = Aggregator.fuse(ranked_lists=results, method=args.fusion, normalization=args.normalization, percentile_distributions=distributions, linear_weights=weights)
    ranked_lists_dict = [
        {doc['corpus_id']: corpus[doc['corpus_id']] for doc in ranked_list if doc['corpus_id'] in corpus}
        for ranked_list in ranked_lists
    ]
    
    #------------------------------------------------------#
    #                       RERANKING
    #------------------------------------------------------#
    if args.run_monobert:
        print(f"{sep}\n# Re-ranking with monoBERT \n{sep}")
        ranked_lists = Ranker.cross_encoder_search(queries, ranked_lists_dict, model_name_or_path=model_ckpts['monobert'][args.models_domain])

    print(ranked_lists)
    #------------------------------------------------------#
    #                       EVALUATION
    #------------------------------------------------------#
    #print(f"{sep}\n# Evaluation \n{sep}")
    #run_evaluation(predictions=[[x['corpus_id'] for x in res] for res in ranked_lists], labels=pos_pids, log2wandb=False, args=args)


if __name__ == '__main__':
    args = argparse.Namespace(
        output_dir='output',
        models_domain='general',  # or 'legal'
        data_split='dev',         # or 'test'
        run_bm25=True,
        run_e5=True,
        run_bge=True,
        run_colbert=True,
        run_dpr=True,
        run_splade=True,
        analyze_score_distributions=False,
        normalization='none',     # or 'percentile-rank', 'normal-curve-equivalent'
        fusion='rrf',
        tune_linear_fusion_weight=False,
        run_monobert=True
    )
    main(args)

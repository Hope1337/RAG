
from src.retrievers.hybrid import Aggregator, Ranker
import pandas as pd
from os.path import exists, join
import numpy as np

class CustomeRetriever():

    def __init__(self, 
        output_dir='output',
        models_domain='general',  # or 'legal'
        run_bm25=True,
        run_e5=True,
        run_bge=True,
        run_colbert=True,
        run_dpr=True,
        run_splade=True,
        run_monobert=True,
        run_baai=True,
        topk_bm25=5,
        topk_e5=5,
        topk_bge=5,
        topk_colbert=5,
        topk_dpr=5,
        topk_splade=5,
        topk_monobert=5,
        topk_baai=5,
        normalization='none',  
        fusion='rrf'):
        
        self.output_dir    = output_dir
        self.models_domain = models_domain

        self.run_bm25     = run_bm25
        self.topk_bm25    = topk_bm25

        self.run_e5       = run_e5
        self.topk_e5      = topk_e5

        self.run_bge      = run_bge
        self.topk_bge     = topk_bge

        self.run_colbert  = run_colbert
        self.topk_colbert = topk_colbert

        self.run_dpr      = run_dpr
        self.topk_dpr     = topk_dpr

        self.run_splade   = run_splade
        self.topk_splade  = topk_splade

        self.run_monobert  = run_monobert
        self.topk_monobert = topk_monobert

        self.run_baai      = run_baai
        self.topk_baai     = topk_baai

        self.normalization = normalization
        self.fusion        = fusion

        self.model_ckpts = {
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
    
    def retrieve(self, corpus, queries):
        results = {}

        if self.run_bm25:
            print(f"\n# Ranking with BM25\n")
            results['bm25'] = Ranker.bm25_search(queries, corpus, return_topk=self.topk_bm25, do_preprocessing=True, k1=2.5, b=0.2)
        
        if self.run_dpr:
            print(f"\n# Ranking with DPR\n")
            results['dpr'] = Ranker.single_vector_search(queries, corpus, return_topk=self.topk_dpr,model_name_or_path=self.model_ckpts['dpr'][self.models_domain])

        if self.run_splade:
            print(f"\n# Ranking with SPLADE\n")
            results['splade'] = Ranker.single_vector_search(queries, corpus, return_topk=self.topk_splade ,model_name_or_path=self.model_ckpts['splade'][self.models_domain])

        if self.run_colbert:
            print(f"\n# Ranking with ColBERT\n")
            results['colbert'] = Ranker.multi_vector_search(queries, corpus, return_topk=self.topk_colbert, model_name_or_path=self.model_ckpts['colbert'][self.models_domain])
        
        if self.run_baai:
            print(f"\n# Ranking with BAAI\n")
            results['baai'] = Ranker.hugging_search(queries, corpus, 'BAAI/bge-large-zh-v1.5', self.topk_baai)

        if self.run_e5:
            print(f"\n# Ranking with e5\n")
            results['e5'] = Ranker.hugging_search(queries, corpus, 'intfloat/e5-large-v2', self.topk_e5)

        
        distributions, weights = {}, {}
        if self.fusion == 'nsf':

            weights = {system: 1/len(results) for system in results}

            if self.normalization == 'percentile-rank' or self.normalization == 'normal-curve-equivalent':
                distr_df = pd.read_csv(join(self.output_dir, f"score_distributions_raw_{self.eval_type}_28k.csv"))
                distributions = {k: np.array(v) for k, v in distr_df.to_dict('series').items()}

        print(f"\n# Fusing results with {self.fusion.upper()}{' ('+self.normalization+')' if self.fusion == 'nsf' else ''}\n")
        ranked_lists = Aggregator.fuse(ranked_lists=results, method=self.fusion, normalization=self.normalization, percentile_distributions=distributions, linear_weights=weights)
    
        #------------------------------------------------------#
        #                       RERANKING
        #------------------------------------------------------#
        if self.run_monobert:
            ranked_lists_dict = [
                {doc['corpus_id']: corpus[doc['corpus_id']] for doc in ranked_list if doc['corpus_id'] in corpus}
                for ranked_list in ranked_lists
            ]
            print(f"\n# Re-ranking with monoBERT \n")
            ranked_lists = Ranker.cross_encoder_search(queries, ranked_lists_dict,return_topk=self.topk_monobert ,model_name_or_path=self.model_ckpts['monobert'][self.models_domain])
        
        return ranked_lists

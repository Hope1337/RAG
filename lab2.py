#from sentence_transformers import SentenceTransformer
#import faiss

#model = SentenceTransformer('intfloat/e5-large-v2')
#input_texts = [
    #"John is a assistant at my work",
    #"Hi there, how are you today",
    #"CDC is stand for Critical Department of Chinsu"
#]
#embeddings = model.encode(input_texts, normalize_embeddings=True)

#print(embeddings.shape[1])
#index = faiss.IndexFlatL2(embeddings.shape[1]) 
#index.add(embeddings)

#queries = [
    #'What is CDC?',
    #'Who is John?'
#]
#embeddings = model.encode(queries, normalize_embeddings=True)

#distances, indices = index.search(embeddings, 2)

#print(distances)
#print(indices)

from src.retrievers.hybrid import Ranker
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

results = Ranker.hugging_search(queries, corpus, model_name='BAAI/bge-large-zh-v1.5')

for result in results:
    print(result) 
    print()


from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
import jieba_hant
from tqdm import tqdm
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import glob, os
import pickle
from sklearn.manifold import SpectralEmbedding


def load_corpus():
    corpuses = [ filename for filename in glob.glob('data/*/*.txt') ]
    corpus = []
    for txt_file in corpuses:
        print(txt_file)
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                corpus.append(line.strip())
    return corpus

def preprocess(texts): # list of texts
    corpus = []
    stop_tokens = []
    with open('data/stopwords.txt', 'r') as f:
        for line in f.readlines():
            stop_tokens.append(line.strip())
    for text in tqdm(texts, dynamic_ncols=True):
        segments = jieba_hant.cut(text)
        segments = list(filter(lambda a: a not in stop_tokens and a != '\n', segments))
        corpus.append(' '.join(segments))
    return corpus


def correlation_matrix(corpus, n_components):
    tfidf = HashingVectorizer(n_features=int(1e4)).fit_transform(corpus)
    print(tfidf.shape)
    correlation_similarities = []
    embeddings = SpectralEmbedding(n_components=n_components, affinity='rbf').fit_transform(tfidf)
    # for idx in tqdm(range(len(corpus)), dynamic_ncols=True):
    #     correlation_similarity = linear_kernel(tfidf[[idx]], tfidf).flatten()
    #     correlation_similarities.append(correlation_similarity)

    # correlation_similarities = np.array(correlation_similarities)
    # print(correlation_similarities.shape)
    # correlation_similarities = np.fill_diagonal(correlation_similarities, 0)
    return correlation_similarities

if __name__ == "__main__":
    if os.path.exists('corpus.pkl'):
        corpus = pickle.load(open('corpus.pkl', 'rb'))
    else:
        corpus = load_corpus()
        corpus = preprocess(corpus)
        with open('corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)

    print('calculate C')
    n_components = 5
    C = correlation_matrix(corpus, n_components)
    kmeans = KMeans(n_clusters=n_components)
    cluster_assignment = kmeans.fit_transform(C)
    print(cluster_assignment[:100])

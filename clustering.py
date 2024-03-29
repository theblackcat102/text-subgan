from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
import jieba_hant
from tqdm import tqdm
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import glob, os
import pickle
from sklearn.manifold import SpectralEmbedding
from utils import chunks
import sklearn

'''
    Initialize the clustering of corpus
'''

def load_corpus():
    corpuses = [ filename for filename in glob.glob('data/*/*.txt') ]
    corpus = []
    for txt_file in corpuses:
        print(txt_file)
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                corpus.append(line.strip())
    return corpus

def load_title(filename='data/kkday_dataset/train_title.txt'):
    corpus = []
    with open(filename, 'r') as f:
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
    
    # print(tfidf.shape)
    correlation_similarities = []
    embeddings = SpectralEmbedding(n_components=n_components, affinity='rbf').fit_transform(corpus)
    # for idx in tqdm(range(len(corpus)), dynamic_ncols=True):
    #     correlation_similarity = linear_kernel(tfidf[[idx]], tfidf).flatten()
    #     correlation_similarities.append(correlation_similarity)

    # correlation_similarities = np.array(correlation_similarities)
    # print(correlation_similarities.shape)
    # correlation_similarities = np.fill_diagonal(correlation_similarities, 0)
    return embeddings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--components', type=int, default=20)
    args = parser.parse_args()

    if os.path.exists('title_corpus.pkl'):
        corpus = pickle.load(open('title_corpus.pkl', 'rb'))
    else:
        corpus = load_title()
        # corpus = preprocess(corpus)
        with open('title_corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)

    print('calculate C : {}'.format(args.components))
    n_components = args.components
    context = []
    tfidf = TfidfVectorizer().fit_transform(corpus)
    for corpus_ in chunks(tfidf, 42000):
        print(corpus_.shape)
        C = correlation_matrix(corpus_, n_components)
        context.append(C)

    context = np.concatenate(context, axis=0)
    ss = sklearn.preprocessing.StandardScaler(copy=False)
    context = ss.fit_transform(context)
    kmeans = KMeans(n_clusters=n_components)
    cluster_assignment = kmeans.fit_predict(context)
    with open('latent_variable_{}.pkl'.format(n_components), 'wb') as f:
        pickle.dump(context, f)

    with open('kmeans_{}.pkl'.format(n_components), 'wb') as f:
        pickle.dump(kmeans, f)

    cluster_latent = {}
    cluster_p = {}
    for idx, text in enumerate(corpus):
        cluster_latent[text] = context[idx]
        cluster_p[text] = cluster_assignment[idx]

    with open('initial_cluster_{}.pkl'.format(n_components), 'wb') as f:
        pickle.dump({
            'p': cluster_p, 'latent': cluster_latent
        }, f)
    print(cluster_assignment[:100])

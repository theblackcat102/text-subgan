from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def load_corpus(filename):
    corpus = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            corpus.append(line.strip().split(' '))
    return corpus

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('reference_file', type=str)
    parser.add_argument('generated_file', type=str)
    args = parser.parse_args()
    generate_corpus = load_corpus(args.generated_file)
    reference_corpus = load_corpus(args.reference_file)
    scores_weights = { str(gram): [1/gram] * gram for gram in range(1, 5)  }
    scores = { str(gram): 0 for gram in range(1, 5)  }

    for ref, gen in zip(reference_corpus, generate_corpus):
        for key, weights in scores_weights.items():
            scores[key] += sentence_bleu([ref], gen, weights, 
                smoothing_function=SmoothingFunction().method5)
    print(scores, len(reference_corpus))
    for key, weights in scores.items():
        scores[key] /= len(reference_corpus)
    print(scores)
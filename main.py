import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import re


def preprocess_corpus(corpus):
    corpus = corpus.lower()
    corpus = re.sub(r"[^a-z\s]", "", corpus)  # keep only alphabets
    words = corpus.split()
    return words

def get_vocab(words):
    vocab = sorted(set(words))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    return vocab, word_to_id, id_to_word


def build_cooccurrence_matrix(words, word_to_id, window_size=4):
    vocab_size = len(word_to_id)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for index, word in enumerate(words):
        word_id = word_to_id[word]
        start = max(0, index - window_size)
        end = min(len(words), index + window_size + 1)
        for i in range(start, end):
            if i != index:
                context_word = words[i]
                context_id = word_to_id[context_word]
                matrix[word_id][context_id] += 1
    return matrix


def reduce_dimensions(matrix, k=2):
    svd = TruncatedSVD(n_components=k)
    reduced = svd.fit_transform(matrix)
    return reduced

# Plot
def plot_embeddings(reduced_matrix, id_to_word, words_to_plot=20):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(list(id_to_word.values())[:words_to_plot]):
        x, y = reduced_matrix[i, 0], reduced_matrix[i, 1]
        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, label, fontsize=9)
    plt.title("2D Word Embeddings from Co-occurrence Matrix")
    plt.show()


if __name__ == "__main__":
    corpus = """
    Artificial intelligence and machine learning are transforming the world.
    Machine learning helps computers learn patterns from data.
    Natural language processing enables computers to understand human language.
    """

    words = preprocess_corpus(corpus)
    vocab, word_to_id, id_to_word = get_vocab(words)

    co_matrix = build_cooccurrence_matrix(words, word_to_id, window_size=4)

    reduced_matrix = reduce_dimensions(co_matrix, k=2)

    plot_embeddings(reduced_matrix, id_to_word, words_to_plot=15)

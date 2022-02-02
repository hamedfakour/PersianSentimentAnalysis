
import os
import pandas as pd
from hazm import Normalizer, word_tokenize
from gensim.models import Word2Vec

base_path = os.path.join(os.path.dirname(__file__), 'data/')

def text_normalizer(df):
    df['title'] = df['title'].fillna('').str.replace('_x000D_', ' ').str.replace(r'[^\w\s]+', '')
    df['comment'] = df['comment'].fillna('').str.replace('_x000D_', ' ').str.replace(r'[^\w\s]+', '')
    normalizer = Normalizer()
    df["title"] = df["title"].apply(lambda txt: normalizer.normalize(txt))
    df['comment'] = df["comment"].apply(lambda txt: normalizer.normalize(txt))
    return df[['title', 'comment']]

def corpus_tokenizer(corpus):
    sentences = []
    for i, v in corpus['title'].iteritems():
        sent = word_tokenize(v)
        sentences.append(sent)
    for i, v in corpus['comment'].iteritems():
        sent = word_tokenize(v)
        sentences.append(sent)
    return sentences


def create_word2vecModel():
    train_clear_file = base_path + 'CI_train_clear.csv'
    corpus = pd.read_csv(train_clear_file)
    corpus = text_normalizer(corpus)
    sentences = corpus_tokenizer(corpus)
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save(base_path + "word2vec.model")

if __name__ == '__main__':
    model = Word2Vec.load(base_path + "word2vec.model")
    print(model.wv['عجیبه'])
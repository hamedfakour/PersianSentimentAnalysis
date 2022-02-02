
from torch import nn, cuda
from collections import Counter
from hazm import Normalizer, word_tokenize
import pandas as pd
import numpy as np


class SentimentLSTM(nn.Module):

    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p=0.5, batch_size=50):
        super().__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab  # number of unique words in vocabulary
        self.n_layers = n_layers  # number of LSTM layers
        self.n_hidden = n_hidden  # number of hidden nodes in LSTM
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_words):
        # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)  # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)  # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(self.batch_size, -1)  # (batch_size, seq_length*n_output)

        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]  # (batch_size, 1)

        return sigmoid_last, h

    def init_hidden(self, batch_size):  # initialize hidden weights (h,c) to 0
        device = "cuda" if cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return h



def merge_data(src_path, tgt_path, type):
    src = pd.read_csv(src_path)
    src['title'] = src['title'].fillna('')
    src['comment'] = src['comment'].fillna('')
    src["text"] = src["title"] + src["comment"]
    if type == 'test':
        tgt = pd.read_csv(tgt_path)
        src = pd.merge(src, tgt, on=["id"], how="inner")
    return src


def text_normalizer(df):
    df['recommend'] = df['recommend'].map({'recommended': 1, 'not_recommended': 0,
                                   'recommended ':1, 'not_recommended ':0})
    normalizer = Normalizer()
    df["text"] = df["text"].str.replace('_x000D_', ' ').str.replace(r'[^\w\s]+', '')
    df['text_normalized'] = df["text"].apply(lambda txt: normalizer.normalize(txt))
    return df


def pad_text(encoded_reviews, seq_length):
    reviews = []
    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0] * (seq_length - len(review)) + review)
    return np.array(reviews)


padded_reviews = pad_text(encoded_reviews, seq_length = 200)
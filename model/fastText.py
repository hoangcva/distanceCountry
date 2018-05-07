# -*- coding: utf-8 -*-
import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models.fasttext import FastText

# path data
pathdata = '../vn_news_country'

# list stopwords
filename = './stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']


def normalize_text(article):
    listpunctuation = string.punctuation
    for i in listpunctuation:
        article = article.replace(i, ' ')
    return article


def remove_stopword(article):
    rearticle = []
    words = article.split()
    for word in words:
        if word not in list_stopwords:
            rearticle.append(word)
    article2 = ' '.join(rearticle)

    return article2


def tokenize(article):
    text_token = ViTokenizer.tokenize(article.decode('utf-8'))
    return text_token


def read_data(path):
    traindata = []
    listfile = os.listdir(path)
    for namefile in listfile:
        with open(pathdata + '/' + namefile) as f:
            datafile = f.read()
        article = tokenize(remove_stopword(normalize_text(datafile))).split()
        traindata.append(article)
    return traindata


if __name__ == '__main__':
    train_data = read_data(pathdata)

    model_fasttext = FastText(size=150, window=10, min_count=2, workers=4, sg=1)
    model_fasttext.build_vocab(train_data)
    model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)

    model_fasttext.wv.save("fasttext_gensim.model")

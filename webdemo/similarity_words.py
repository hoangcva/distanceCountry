# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import spacy
from spacy.tokens import Doc

pathcountry = '../model/country'

with open(pathcountry, 'r') as f:
    listcountry = f.readlines()
    listcountry = [country.strip().decode('utf-8') for country in listcountry]


def getvector(namemethod):
    words_np = []
    words_label = []

    if namemethod == 'word2vec_skipgram':
        model = KeyedVectors.load('../model/word2vec_skipgram.model')
        for word in model.vocab.keys():
            if word in listcountry:
                words_np.append(model[word])
                words_label.append(word)
    elif namemethod == 'word2vec_cbow':
        model = KeyedVectors.load('../model/word2vec_cbow.model')
        for word in model.vocab.keys():
            if word in listcountry:
                words_np.append(model[word])
                words_label.append(word)
    elif namemethod == 'Spacy':
        nlp = spacy.load('vi_core_news_md')
        tokens = Doc(nlp.vocab, words=listcountry)
        for word in tokens:
            if word.has_vector:
                words_np.append(word.vector)
                words_label.append(word.text)
    elif namemethod == 'fastText':
        model = KeyedVectors.load('../model/fasttext_gensim.model')
        for word in model.vocab.keys():
            if word in listcountry:
                words_np.append(model[word])
                words_label.append(word)
    else:
        pass
        # print "No"

    # words_np = np.array(words_np).reshape(1, -1)
    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    list_vector_country = {}
    for index, vec in enumerate(reduced):
        list_vector_country[words_label[index]] = vec

    return list_vector_country


def list_country(namemethod = 'word2vec_skipgram'):
    list_vector_country = getvector(namemethod)
    return sorted(list_vector_country.keys())


def distance_country_euclidean(word1, word2, namemethod = 'word2vec_skipgram'):
    list_vector_country = getvector(namemethod)
    print namemethod
    vec_word1 = list_vector_country[word1]
    vec_word2 = list_vector_country[word2]

    print'Distance from %s to %s: %f' % (word1, word2, np.sqrt(sum((vec_word1 - vec_word2) ** 2)))

    return str(round(np.sqrt(sum((vec_word1 - vec_word2) ** 2)), 4))

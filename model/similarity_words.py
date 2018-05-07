# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np

model = KeyedVectors.load("./country.model")

pathcountry = './country'

with open(pathcountry, 'r') as f:
    listcountry = f.readlines()
    listcountry = [country.strip().decode('utf-8') for country in listcountry]

words_np = []
words_label = []

for word in model.vocab.keys():
    if word in listcountry:
        words_np.append(model[word])
        words_label.append(word)
pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)

list_vector_country = {}
for index, vec in enumerate(reduced):
    list_vector_country[words_label[index]] = vec


def distance_country_euclidean(word1, word2):
    vec_word1 = list_vector_country[word1.decode('utf-8')]
    vec_word2 = list_vector_country[word2.decode('utf-8')]

    print'Distance from %s to %s: %f' % (word1, word2, np.sqrt(sum((vec_word1 - vec_word2) ** 2)))

    return np.sqrt(sum((vec_word1 - vec_word2) ** 2))


(distance_country_euclidean('Việt_Nam', 'Ukraine'))
(distance_country_euclidean('Việt_Nam', 'Brazil'))

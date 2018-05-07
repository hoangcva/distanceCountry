import pandas as pd
from similarity_words import distance_country_euclidean

test = pd.read_csv('test.csv', encoding='utf-8')

country1 = test['c1']
country2 = test['c2']
country3 = test['c3']

accuracy = 0

for i in range(len(country1)):
    l1 = float(distance_country_euclidean(country1[i], country2[i], namemethod = 'fastText'))
    l2 = float(distance_country_euclidean(country1[i], country3[i], namemethod = 'fastText'))
    if( l1 < l2):
        accuracy += 1
        print l1, l2, accuracy

accuracy = 1.0 * accuracy/len(country1) *100

print accuracy

# accuracy word2vec_skipgram: 97.7272727273
# accuracy word2vec_cbow: 70.4545454545
# accuracy Spacy: 59.0909090909
# accuracy fastText: 99.025974026

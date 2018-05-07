import gensim.models.keyedvectors as word2vec
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.offline import plot
import spacy
from spacy.tokens import Doc
import numpy as np

pathcountry = '../model/country'
with open(pathcountry, 'r') as f:
    listcountry = f.readlines()
    listcountry = [country.strip().decode('utf-8') for country in listcountry]


def visualize(namemethod='word2vec_skipgram'):
    words_np = []
    words_label = []

    if namemethod == 'word2vec_skipgram':
        model = word2vec.KeyedVectors.load('../model/word2vec_skipgram.model')
        for word in model.vocab.keys():
            if word in listcountry:
                words_np.append(model[word])
                words_label.append(word)
    elif namemethod == 'word2vec_cbow':
        model = word2vec.KeyedVectors.load('../model/word2vec_cbow.model')
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
        model = word2vec.KeyedVectors.load('../model/fasttext_gensim.model')
        for word in model.vocab.keys():
            if word in listcountry:
                words_np.append(model[word])
                words_label.append(word)
    else:
        pass
        # print "No"

    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    x = []
    y = []
    namecountry = []

    for index, vec in enumerate(reduced):
        x.append(vec[0])
        y.append(vec[1])
        namecountry.append(words_label[index])
    trace = go.Scatter(
        x=x,
        y=y,
        text=namecountry,
        mode='markers+text',
        textposition='top'
    )
    data = [trace]
    layout = go.Layout(
        showlegend=False,
        height=700,
        width=1000,
    )
    fig = go.Figure(data=data, layout=layout)
    my_plot_div = plot(fig, output_type='div')

    return my_plot_div
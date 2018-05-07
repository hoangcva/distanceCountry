import gensim.models.keyedvectors as word2vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = word2vec.KeyedVectors.load('./fasttext_gensim.model')

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


def visualize():
    fig, ax = plt.subplots()

    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]

        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y))

    plt.show()
    return


if __name__ == '__main__':
    visualize()

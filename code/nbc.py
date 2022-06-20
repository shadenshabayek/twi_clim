import nltk
import string

from itertools import chain
from nltk.classify import NaiveBayesClassifier as nbc
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist


def get_documents(lim_word_features):

    mr = CategorizedPlaintextCorpusReader('./data/twi_clim_corpus', r'(?!\.).*\.txt',
                                          cat_pattern=r'(neg|neu|pos)/.*')
    stop = stopwords.words('english')
    documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i in mr.fileids()]
    word_features = FreqDist(chain(*[i for i,j in documents]))
    word_features = list(word_features.keys())[0:lim_word_features]

    return word_features

def train(number_features, word_features):

    numtrain = int(len(documents) * 90 / 100)
    train_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in documents[:numtrain]]
    test_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag  in documents[numtrain:]]

    classifier = nbc.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(number_features)


def main():

    word_features = get_documents(lim_word_features = 10000)
    number_features = 10
    train(number_features, word_features)

if __name__ == '__main__':
    main()

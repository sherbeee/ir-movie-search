
import math
import pickle
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import ast

from os.path import exists


NUMBER_OF_RESULTS = 10


class BM25Okapi():
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.calculate_corpus(corpus)

    def set_title_mapping(self, titles):
        self.titles = titles

    def calculate_corpus(self, corpus):
        self.corpus_size = len(corpus)
        total_token_count = 0  # total token count across corpus
        self.doc_len = []  # length of each doc
        self.tf = []  # document frequencies (term frequency for each document)

        for document in corpus:
            self.doc_len.append(len(document))
            total_token_count += len(document)

            tf = self.tf_(document)
            self.tf.append(tf)

        self.df = self.df_(corpus)

        self.average_doc_length = total_token_count / self.corpus_size

        self.idf = self.idf_(self.df, self.corpus_size)

        print('done??')

    def tf_(self, doc):
        tf = {}

        for term in doc:
            if term in tf.keys():
                tf[term] += 1
            else:
                tf[term] = 1

        return tf

    def df_(self, corpus):
        df = {}

        for doc in corpus:
            term_seen = {}
            for term in doc:
                if term not in term_seen.keys():
                    if term not in df.keys():
                        df[term] = 1
                    else:
                        df[term] += 1

        return df

    def idf_(self, df, corpus_size):

        idf = {}

        for term, freq in df.items():
            idf[term] = round(math.log(corpus_size/freq), 2)

        return idf

    def get_scores(self, tokens):

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for term in tokens:
            if term not in self.idf.keys():
                continue
            term_freq = np.array(
                [doc[term] if term in doc.keys() else 0 for doc in self.tf])
            score += (self.idf[term] * ((self.k1 + 1) * term_freq /
                                        (self.k1 * ((1 - self.b) + self.b * (doc_len / self.average_doc_length)) + term_freq)))
        return score

    def query(self, query):
        # tokenize
        tokens = word_tokenize(query)

        # stop words
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.lower(
        ) not in stop_words and t.lower() not in string.punctuation]

        # stemming
        ps = PorterStemmer()
        tokens = [ps.stem(t) for t in tokens]

        # get scores
        results = self.get_scores(tokens)

        ranked_titles = [{"title": self.titles[i], "score":score}
                         for i, score in enumerate(results)]

        ranked_titles.sort(
            key=lambda x: x['score'], reverse=True)

        return ranked_titles[:NUMBER_OF_RESULTS]


file_exists = exists('BM25.pickle')

if file_exists:
    file = open("BM25.pickle", "rb")
    model = pickle.load(file)
else:
    # load csv.
    movie_data = pd.read_csv('parsed_data.csv', header=0)
    raw_list = movie_data.values.tolist()

    corpus = []
    titles = []
    for row in raw_list:
        titles.append(row[1])
        corpus.append(ast.literal_eval(row[2]))

    model = BM25Okapi(corpus)
    model.set_title_mapping(titles)

    # save to pickle
    file = open("BM25.pickle", "wb")
    pickle.dump(model, file)


print("model loaded.")
while True:
    query_input = input("Please enter a query.")
    start_time = time.time()
    results = model.query(query_input)
    print("results - ", results)
    print("time taken - ", (time.time() - start_time), 'seconds')

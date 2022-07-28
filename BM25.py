
import math
import numpy as np


class BM25Okapi():
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.calculate_corpus(corpus)

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
            if term not in self.tf.keys() or self.idf.keys():
                continue
            term_freq = np.array([doc[term] for doc in self.tf])
            score += (self.idf[term] * ((self.k1 + 1) * term_freq /
                                        (self.k1 * ((1 - self.b) + self.b * (doc_len / self.avgdl)) + term_freq)))
        return score


# TODO: put in parsed_data.csv as corpus
# TODO: add mapping of doc number to movie title
# TODO: return ranking.
BM25Okapi()

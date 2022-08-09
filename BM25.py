
import math
import pickle
from pyexpat import model
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import ast
from sentence_transformers import SentenceTransformer, util
import re

from os.path import exists


NUMBER_OF_RESULTS = 10
NUMBER_OF_CANDIDATES = 50


class BM25Okapi():
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.calculate_corpus(corpus)

    def set_title_mapping(self, titles):
        self.titles = titles

    def set_plots_mapping(self, plots):
        self.plots = plots

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

    def print_results(self, results, time):
        print('\n-----------------Search Results-----------------')
        print('Time Taken: {}'.format(time))
        # print('User Query: {}'.format(query))
        for i in range(len(results)):
            print('{}. {} ----- Score: {}'.format(i+1,
                  results[i]['title'].strip(), results[i]['score']))
        print('------------------------------------------------\n')

    def query(self, query, top_k):
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

        ranked_titles = [{"title": self.titles[i], "plot": self.plots[i], "score":score}
                         for i, score in enumerate(results)]

        ranked_titles.sort(
            key=lambda x: x['score'], reverse=True)

        return ranked_titles[:top_k]


class BM25WithBertRerank():

    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        # create encoder
        self.encoder = SentenceTransformer(model_name)

        file_exists = exists('BM25.pickle')

        if file_exists:
            file = open("BM25.pickle", "rb")
            self.model = pickle.load(file)
        else:
            # load csv.
            movie_data = pd.read_csv('parsed_data.csv', header=0)
            plots = list(pd.read_csv('movie_data.csv', header=0)
                         ['Plot'].to_numpy())
            raw_list = movie_data.values.tolist()

            corpus = []
            titles = []
            for row in raw_list:
                titles.append(row[1])
                corpus.append(ast.literal_eval(row[2]))

            model = BM25Okapi(corpus)
            model.set_title_mapping(titles)
            model.set_plots_mapping(plots)

            # save to pickle
            file = open("BM25.pickle", "wb")
            pickle.dump(model, file)

            self.model = model

    def process_plot(self, plots):
        processed_plots = []

        for plot in plots:
            processed_plot = re.sub(r'\[.*?\]', '', plot)
            processed_plots.append(processed_plot)

        return processed_plots

    def cosine_sim(self, query, plots):
        scores = util.cos_sim(query, plots).tolist()[0]
        return scores

    def rerank(self, query, results, top_k):
        # encode query
        query_emb = self.encoder.encode([query])

        # create embeddings for plots
        plots = self.process_plot([r['plot'] for r in results])
        plots_emb = self.encoder.encode(plots)

        # compute cosine similarity between query and each plot
        scores = self.cosine_sim(query_emb, plots_emb)

        # update the scores
        for i in range(len(scores)):
            results[i]['score'] = scores[i]

        # re-sort the results
        results.sort(
            key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def search(self, query, top_k, rerank=False, print_results=True):
        start_time = time.time()

        if rerank:
            results = self.model.query(query, NUMBER_OF_CANDIDATES)
            results = self.rerank(query, results, top_k)
        
        else:
            results = self.model.query(query, top_k)

        if print_results:
            self.print_results(results, time.time() - start_time, rerank)

        return results

    def print_results(self, results, time, rerank):
        if rerank:
            print('\n------------Reranked Search Results-------------')
        else:
            print('\n-----------------Search Results-----------------')
        print('Time Taken: {}'.format(time))
        # print('User Query: {}'.format(query))
        for i in range(len(results)):
            print('{}. {} ----- Score: {}'.format(i+1,
                  results[i]['title'].strip(), results[i]['score']))
        print('------------------------------------------------\n')


def inititalise_BM25(model_path='BM25.pickle'):
    file_exists = exists(model_path)

    if file_exists:
        file = open(model_path, "rb")
        print("FILE FOUND and OPENED")
        model = pickle.load(file)
        print("THIS IS INITIALISED")
    else:
        # load csv.
        movie_data = pd.read_csv('parsed_data.csv', header=0)
        plots = list(pd.read_csv('movie_data.csv', header=0)
                     ['Plot'].to_numpy())
        raw_list = movie_data.values.tolist()

        corpus = []
        titles = []
        for row in raw_list:
            titles.append(row[1])
            corpus.append(ast.literal_eval(row[2]))

        model = BM25Okapi(corpus)
        model.set_title_mapping(titles)
        model.set_plots_mapping(plots)

        # save to pickle
        file = open(model_path, "wb")
        pickle.dump(model, file)

    print("model loaded.")
    return model

# selected_engine = input("Please select if you want to run BM25 or BM25 with BERT Reranking. Select 1 for BM25 and 2 for BM25 with BERT Reranking:\n")


# if int(selected_engine) == 2:
#     search_engine = BM25WithBertRerank()

#     while True:
#         query_input = input("Please enter a query.\nQuery: ")
#         results = search_engine.search(query_input, NUMBER_OF_RESULTS)
#         results = search_engine.search(
#             query_input, NUMBER_OF_RESULTS, rerank=True)

# else:
#     model = inititalise_BM25()

#     while True:
#         query_input = input("Please enter a query.\nQuery: ")
#         start_time = time.time()
#         results = model.query(query_input, NUMBER_OF_RESULTS)
#         # print("results - ", results)
#         # print("time taken - ", (time.time() - start_time), 'seconds')
#         model.print_results(results, time.time() - start_time)

from flask import Flask
import pandas as pd
from sqlalchemy import null
from BM25 import BM25Okapi
from BM25 import *
from semantic_search.semantic_search import SearchUsingBert
from flask import request
app = Flask(__name__)

K_RESULTS = 5

semantic_search_engine = None
bm25_bert_search_engine = None
bm25_search_engine = None
"""
GET params:

engine: specifies which search implementation to use.
        BERT      = SearchUsingBert
        BM25      = BM25
        BM25_BERT = BM25WithBertRerank

query: user's query for movie plot description.
"""


@app.route("/")
def home():
    return "Hello World!"


@app.route("/search", methods=['GET'])
def search():
    requested_engine = request.args.get("engine")
    query = request.args.get("query")

    if requested_engine == "BERT":
        results = semantic_search_engine.search(query, K_RESULTS)
    elif requested_engine == "BM25":
        results = bm25_search_engine.query(query, K_RESULTS)
    else:
        results = bm25_bert_search_engine.search(query, K_RESULTS)

    return {"results":results}


if __name__ == "__main__":

    movie_data = pd.read_csv('movie_data.csv', header=0)
    search_engine = SearchUsingBert(movie_data, training_data=None , model_file_path="../semantic_search/bert_models/search-base-bert-model", emb_file_path="../semantic_search/embeddings/plot_embeddings_base.pkl", finetune=False)

    with open('BM25.pickle', 'rb') as file:
        bm25_search_engine = pickle.load(file)

    bm25_bert_search_engine = BM25WithBertRerank()
    app.run(debug=True)

from flask import Flask
from flask_cors import CORS
import pandas as pd
from sqlalchemy import null
from BM25 import BM25Okapi
from BM25 import *
from semantic_search.semantic_search import SearchUsingBert
from flask import request
import json
import numpy as np

app = Flask(__name__)
CORS(app)

K_RESULTS = 5

semantic_search_engine = None
bm25_bert_search_engine = None
bm25_search_engine = None


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


"""
GET params:

engine: specifies which search implementation to use.
        BERT      = SearchUsingBert
        BM25      = BM25
        BM25_BERT = BM25WithBertRerank

query: user's query for movie plot description.
"""


@app.route("/test", methods=['GET'])
def home():
    requested_engine = request.args.get("engine")
    query = request.args.get("query")
    res = " Hello World! "
    return {"results": {"query": query, "engine": requested_engine}}


@app.route("/search", methods=['GET'])
def search():
    requested_engine = request.args.get("engine")
    query = request.args.get("query")

    if requested_engine == "BERT":
        results = semantic_search_engine.search(query, K_RESULTS)
    elif requested_engine == "BM25":
        results = bm25_search_engine.query(query, K_RESULTS)
    else:
        results = bm25_bert_search_engine.search(query, K_RESULTS, rerank=True)

    results = json.dumps(results, cls=NpEncoder)
    results = json.loads(results)
    return {"results": results}


if __name__ == "__main__":

    movie_data = pd.read_csv('movie_data.csv', header=0)
    semantic_search_engine = SearchUsingBert(movie_data, training_data=None, model_file_path="./semantic_search/search-bert-model",
                                             emb_file_path="./semantic_search/plot_embeddings.pkl", finetune=False)

    with open('BM25.pickle', 'rb') as file:
        bm25_search_engine = pickle.load(file)

    bm25_bert_search_engine = BM25WithBertRerank()
    app.run(debug=True)

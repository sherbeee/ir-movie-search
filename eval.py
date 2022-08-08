import json
from BM25 import *
from semantic_search.semantic_search import SearchUsingBert

"""
Constants
"""
K_RESULTS = 5

"""
Helper functions
"""
def count_relevant_results(query_results,relevant_documents):
    count_relevant = 0
    for query_result in query_results:
        count_relevant = count_relevant + 1 if query_result['title'] in relevant_documents else count_relevant
    return count_relevant

movie_data = pd.read_csv('movie_data.csv', header=0)
semantic_search_engine = SearchUsingBert(movie_data, training_data=None, model_file_path="./semantic_search/bert_models/search-base-bert-model",                                            emb_file_path="./semantic_search/embeddings/plot_embeddings_base.pkl", finetune=False)
with open('BM25.pickle', 'rb') as file:
    bm25_search_engine = pickle.load(file)

bm25_bert_search_engine = BM25WithBertRerank()
test_queries_f = open('relevant_test_queries.json')
test_queries = json.load(test_queries_f)

recall_bm25 = 0
precision_bm25 = 0
recall_bert = 0
precision_bert = 0
recall_bm25_bert = 0
precision_bm25_bert = 0

for query, relevant_titles in test_queries.items():
    bm25_results = bm25_search_engine.query(query,K_RESULTS)
    num_rel_doc = count_relevant_results(bm25_results,relevant_titles)
    recall_bm25 += (num_rel_doc / len(relevant_titles))
    precision_bm25 += (num_rel_doc / K_RESULTS)

    bert_results = semantic_search_engine.search(query,K_RESULTS)
    num_rel_doc = count_relevant_results(bert_results,relevant_titles)
    recall_bert += (num_rel_doc / len(relevant_titles))
    precision_bert += (num_rel_doc / K_RESULTS)

    bm25_bert_results = bm25_bert_search_engine.search(query,K_RESULTS)
    num_rel_doc = count_relevant_results(bm25_bert_results,relevant_titles)
    recall_bm25_bert += (num_rel_doc / len(relevant_titles))
    precision_bm25_bert += (num_rel_doc / K_RESULTS)

# normalize
recall_bm25 /= len(test_queries)
precision_bm25 /= len(test_queries)
recall_bert /= len(test_queries)
precision_bert /= len(test_queries)
recall_bm25_bert /= len(test_queries)
precision_bm25_bert /= len(test_queries)

print("--------- Calculated F score ---------")
print("BM25:               " + str(2 / ( 1/recall_bm25 + 1/precision_bm25)))
print("Bert:               " + str(2 / ( 1/recall_bert + 1/precision_bert)))
print("BM25 + Bert rerank: " + str(2 / ( 1/recall_bm25_bert + 1/precision_bm25_bert)))
print("--------------------------------------")
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


def count_relevant_results(query_results, relevant_documents):
    count_relevant = 0
    for query_result in query_results:
        count_relevant = count_relevant + \
            1 if query_result['title'] in relevant_documents else count_relevant
    return count_relevant


def get_mean_avg_precision(query_results, relevant_documents):
    count_relevant = 0
    precision_at_k = []

    for i, query_result in enumerate(query_results):
        if i + 1 == len(relevant_documents):
            break
        count_relevant = count_relevant + \
            1 if query_result['title'] in relevant_documents else count_relevant
        precision_at_k.append((count_relevant / (i + 1)))

    return sum(precision_at_k) / len(relevant_documents)


movie_data = pd.read_csv('movie_data.csv', header=0)
semantic_search_engine = SearchUsingBert(movie_data, training_data=None, model_file_path="./semantic_search/bert_models/search-bert-model",
                                         emb_file_path="./semantic_search/embeddings/plot_embeddings.pkl", finetune=False)
with open('BM25.pickle', 'rb') as file:
    bm25_search_engine = pickle.load(file)

bm25_bert_search_engine = BM25WithBertRerank()
test_queries_f = open('relevant_test_queries.json', encoding="utf8")
test_queries = json.load(test_queries_f)

recall_bm25 = 0
precision_bm25 = 0
recall_bm25_bert = 0
precision_bm25_bert = 0

recall_bert = 0
precision_bert = 0
recall_bert_rerank = 0
precision_bert_rerank = 0

map_bm25 = 0
map_bert = 0
map_bm25_bert = 0
map_bert_rerank = 0
for query, relevant_titles in test_queries.items():
    bm25_results = bm25_search_engine.query(query, K_RESULTS)
    num_rel_doc = count_relevant_results(bm25_results, relevant_titles)
    recall_bm25 += (num_rel_doc / len(relevant_titles))
    precision_bm25 += (num_rel_doc / K_RESULTS)
    map_bm25 += get_mean_avg_precision(bm25_results, relevant_titles)


    bm25_bert_results = bm25_bert_search_engine.search(
        query, K_RESULTS, print_results=False, rerank=True)
    num_rel_doc = count_relevant_results(bm25_bert_results, relevant_titles)
    recall_bm25_bert += (num_rel_doc / len(relevant_titles))
    precision_bm25_bert += (num_rel_doc / K_RESULTS)
    map_bm25_bert += get_mean_avg_precision(bm25_bert_results, relevant_titles)

    bert_results = semantic_search_engine.search(query, K_RESULTS)
    num_rel_doc = count_relevant_results(bert_results, relevant_titles)
    recall_bert += (num_rel_doc / len(relevant_titles))
    precision_bert += (num_rel_doc / K_RESULTS)
    map_bert += get_mean_avg_precision(bert_results, relevant_titles)

    bert_rerank_results = semantic_search_engine.search(query, K_RESULTS,re_rank=True,rerank_method="cross-encoder")
    num_rel_doc = count_relevant_results(bert_rerank_results, relevant_titles)
    recall_bert_rerank += (num_rel_doc / len(relevant_titles))
    precision_bert_rerank += (num_rel_doc / K_RESULTS)
    map_bert_rerank += get_mean_avg_precision(bert_rerank_results, relevant_titles)


# normalize
recall_bm25 /= len(test_queries)
precision_bm25 /= len(test_queries)
recall_bm25_bert /= len(test_queries)
precision_bm25_bert /= len(test_queries)
recall_bert /= len(test_queries)
precision_bert /= len(test_queries)
recall_bert_rerank /= len(test_queries)
precision_bert_rerank /= len(test_queries)


map_bm25 /= len(test_queries)
map_bm25_bert /= len(test_queries)
map_bert /= len(test_queries)
map_bert_rerank /= len(test_queries)

print("--------- Calculated F score ---------")
print("BM25:               " + str(2 / (1/recall_bm25 + 1/precision_bm25)))
print("BM25 + Bert:        " + str(2 / (1/recall_bm25_bert + 1/precision_bm25_bert)))
print("Bert:               " + str(2 / (1/recall_bert + 1/precision_bert)))
print("Bert rerank:        " + str(2 / (1/recall_bert_rerank + 1/precision_bert_rerank)))

print("--------------------------------------")

print("--------- Calculated Mean Average Precision ---------")
print("BM25:               " + str(map_bm25))
print("BM25 + Bert:        " + str(map_bm25_bert))
print("Bert:               " + str(map_bert))
print("Bert rerank:        " + str(map_bert_rerank))

print("-----------------------------------------------------")

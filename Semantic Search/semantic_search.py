from sentence_transformers import SentenceTransformer, InputExample, models, datasets, losses, util, CrossEncoder
import pandas as pd
from os import path
import time
import re
import pickle
from tqdm import tqdm

class SearchUsingBert():
    
    def __init__(self, data, training_data=None, base_model='sentence-transformers/msmarco-distilbert-base-dot-prod-v3', model_file_path='../Semantic Search/bert_models/search-bert-model', emb_file_path = '../Semantic Search/embeddings/plot_embeddings.pkl', finetune=True, num_of_epochs=3, save_model_path='../Semantic Search/bert_models/search-bert-model'):
        print("Initialising Bert Search Engine...\n")
        
        self.movie_data = data
        self.titles = list(data['Title'].to_numpy())
        self.plots = self.process_plot(list(data['Plot'].to_numpy()))

        # if model file exists, load in the model
        if path.exists(model_file_path):
            print("Model found, loading model from path...")
            self.model = self.load_model(model_file_path)
            print('Model loaded.\n')

        else:
            if training_data == None:
                print("Please provide training data.")
                return
            self.model = self.finetune_model(training_data, finetune, base_model, num_of_epochs, model_file_path)
            
        # if embeddings file exist, load in the embeddings
        if path.exists(emb_file_path):
            print("Pre-computed embeddings for movie plots found, loading from file...")
            
            with open(emb_file_path, "rb") as file:
                emb_data = pickle.load(file)
                self.plots_emb = emb_data['embeddings']
                
            print("Embeddings loaded.\n")

        else:
            print("Pre-computed embeddings for movie plots not found, encoding in progress...")
            self.plots_emb = self.model.encode(self.plots)
            
            # Save pre-computed plot embeddings to file for easier retrieval when testing
            with open(emb_file_path, "wb") as file:
                pickle.dump({'embeddings': self.plots_emb}, file)
                
            print("Embeddings generated.\n")

        print("Initialisation of Bert Search Engine completed.\n")

    def load_model(self, model_file_path=None):
        return SentenceTransformer(model_file_path)

    def process_plot(self, plots):
        processed_plots = []
        
        for plot in plots:
            processed_plot = re.sub(r'\[.*?\]', '', plot)
            processed_plots.append(processed_plot)
        
        return processed_plots

    def finetune_model(self, training_data, finetune, base_model, num_of_epochs, save_model_path):
        print("Model not found, default model will undergo finetuning...")

        if base_model != "sentence-transformers/msmarco-distilbert-base-dot-prod-v3":
            print("User-defined model: {}".format(base_model))
        else:
            print("Default model: {}".format(base_model))
        
        # Create sentence transformer model
        word_embeddings = models.Transformer(base_model)
        pooling = models.Pooling(word_embeddings.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embeddings, pooling])

        if finetune:
            # MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
            # and trains the model so that is is suitable for semantic search
            train_loss = losses.MultipleNegativesRankingLoss(model)

            # For the MultipleNegativesRankingLoss, it is important
            # that the batch does not contain duplicate entries, i.e.
            # no two equal queries and no two equal paragraphs.
            # To ensure this, we use a special data loader
            train_dataloader = datasets.NoDuplicatesDataLoader(training_data, batch_size=8)

            # Tune the model
            warmup_steps = int(len(train_dataloader) * num_of_epochs * 0.1)
            model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_of_epochs, warmup_steps=warmup_steps, show_progress_bar=True)
            
            print('Model finetuning completed.\n')

        # Save the model
        model.save(save_model_path)
        print('Model saved.\n')
        
        return model

    def cosine_sim(self, query, plots):
        scores = util.cos_sim(query, plots).tolist()[0]
        return scores
    
    def get_k_highest(self, k, scores):
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return sorted_scores[0:k]

    def retrieve_movies(self, scores):
        results = []
        for idx, score in scores:
            results.append((self.movie_data.iloc[idx], score))
        
        return results
    
    def re_rank(self, query, scores, rerank_method='cross-encoder', encoder_name='cross-encoder/ms-marco-TinyBERT-L-6'):
        if rerank_method == 'cross-encoder':
            indexes = [i for i, _ in scores]
            inputs = [[query, self.plots[i]] for i in indexes]
            
            cross_encoder = CrossEncoder(encoder_name, max_length=512)
            scores = cross_encoder.predict(inputs)
            
            new_scores = sorted([(indexes[i], scores[i]) for i in range(len(indexes))], key=lambda x: x[1], reverse=True)
            return new_scores
    
    def search(self, query, top_k, print_progress=True, print_results=True, re_rank=False, rerank_method=None, cross_encoder=None):
        if print_progress:
            print('Searching in progress...')
        start_time = time.time()
        query_embedding = self.model.encode([query])
        sim_scores = self.cosine_sim(query_embedding, self.plots_emb)
        highest_k = self.get_k_highest(top_k, sim_scores)
        time_taken = time.time() - start_time
        if print_progress:
            print('Search completed.\n')

        if re_rank:
            if rerank_method == None:
                print('Please specify a re-ranking method.')
                return
            if rerank_method == 'cross-encoder':
                print('Reranking in progress...')
                if cross_encoder == None:
                    highest_k = self.re_rank(query, highest_k, rerank_method)
                else:
                    highest_k = self.re_rank(query, highest_k, rerank_method, cross_encoder)
                print('Reranking completed.')
        
        movies = self.retrieve_movies(highest_k)

        results = []
        for movie, score in movies:
            results.append({
                'title': movie['Title'].strip(),
                'plot': movie['Plot'],
                'score': score,
                'year': movie['Release Year']
            })
        
        if print_results:
            self.print_results(query, time_taken, results)
        
        return results

    def print_results(self, query, time, results):
        print('-----------------Search Results-----------------')
        print('Total Search Time: {}'.format(time))
        print('User Query: {}'.format(query))
        print('\nResults:')
        for i in range(len(results)):
            r = results[i]
            print('{}. {} ({}) ----- Score: {}'.format(i+1, r['title'], r['year'], r['score']))
        print('\n')

# Load in movie data
movie_data = pd.read_csv('../movie_data.csv', header=0)

# Start up the search engine (default - finetuned using basic T5 Model)
search_engine = SearchUsingBert(movie_data)


# Finetuned using generated queries from T5 One Line Summary Model
# search_engine = SearchUsingBert(movie_data, model_file_path="../Semantic Search/bert_models/search-bert-model-2", emb_file_path="../Semantic Search/embeddings/plot_embeddings_2.pkl")

# Base Model
# search_engine = SearchUsingBert(movie_data, training_data, model_file_path="../Semantic Search/bert_models/search-base-bert-model", emb_file_path="../Semantic Search/embeddings/plot_embeddings_base.pkl", finetune=False)

test_query = "spider man and his girlfriend"
k = 5

results = search_engine.search(test_query, k)

# Re-Ranking using Cross-Encoder
reranked_results = search_engine.search(test_query, k, re_rank=True, rerank_method='cross-encoder')
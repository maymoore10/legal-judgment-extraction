# scp -r  morm@192.168.90.212:/home/morm/LegalVerdict_FinalProject/src/unsupervised ~/Downloads/final_project_backup/src/

import re
import csv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from utils.utils import *


models = [
    # {'model_name': 'alephbert-bm', 'model': SentenceTransformer('imvladikon/sentence-transformers-alephbert')},
    #       {'model_name': 'legal_hebert-bm', 'model': SentenceTransformer('avichr/Legal-heBERT')},
    #       {'model_name': 'mbert-base-bm', 'model': SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')},
    #       {'model_name': 'google-mbert-bm', 'model': SentenceTransformer('google-bert/bert-base-multilingual-cased')},
          {'model_name': 'mt5-small', 'model': SentenceTransformer('google/mt5-small')},
          # {'model_name': 't5_base-bm', 'model': SentenceTransformer('sentence-transformers/sentence-t5-base')}
]

# legal_hebert_model = SentenceTransformer('google/mt5-small'),
def summarize(model, text):
    sentences, embeddings = segment_and_embed(model, text)
    clustered_sentences = cluster_sentences(sentences, embeddings)
    summary = summarize_verdicts(clustered_sentences)
    last_sentences_1 = get_last_n_sentences(summary, 1)
    last_sentences_2 = get_last_n_sentences(summary, 2)
    return summary, last_sentences_1, last_sentences_2

path = test_dir
for model_obj in models:
    model_name = model_obj['model_name']
    model = model_obj['model']
    summaries_df = []
    summaries_1last = []
    summaries_2last = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        # Check if it is a file
        if os.path.isfile(file_path):
            # You can open and read the file, or do other processing here
            # For example, to read and print each line of the file:
            with open(file_path, 'r') as file:
                text = file.read().rstrip()
                summary, last_sentences_1, last_sentences_2 = summarize(model, text)
                summaries_df.append({'filename': filename, 'result': summary})
                summaries_1last.append({'filename': filename, 'result': last_sentences_1})
                summaries_2last.append({'filename': filename, 'result': last_sentences_2})
            # print(f"filename: {filename}, summary: {summary}")
        save_csv_file(os.path.join(results_path, f"{today_format}/test/summaries_{model_name}.csv"), summaries_df)
        save_csv_file(os.path.join(results_path, f"{today_format}/test/summaries_1last_{model_name}.csv"), summaries_1last)
        save_csv_file(os.path.join(results_path, f"{today_format}/test/summaries_2last_{model_name}.csv"), summaries_2last)




from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForMaskedLM, BertModel, MT5Tokenizer, MT5ForQuestionAnswering, AutoModel, \
    MT5Model
from bert_score import BERTScorer
import numpy as np
import pandas as pd
import os
from utils.utils import *
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

models = [
    {
        'model_name': 'mbert-bm',
        'tokenizer': BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased'),
        'model': BertModel.from_pretrained('google-bert/bert-base-multilingual-cased'),
        'question': questions[0]

    },
    {
        'model_name': 'aleph-bert-bm',
        'tokenizer': AutoTokenizer.from_pretrained("onlplab/alephbert-base"),
        'model': AutoModel.from_pretrained("onlplab/alephbert-base"),
        'question': questions[0]

    }
]


def bert_score(references, candidates, model, tokenizer):
    evaluations = []
    for (reference, candidate) in zip(references, candidates):
        inputs1 = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)

        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

        similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        evaluations.append(similarity[0][0])
    if len(evaluations) == 0:
        return 0
    return statistics.mean(evaluations)



reference_path = annotated_sentences_test
reference_df = pd.read_csv(reference_path, sep='\t', encoding="utf8")[['filename', 'label']]

path = '/home/morm/legal-IR/results/final/test'
for m in models:
    model = m['model']
    tokenizer = m['tokenizer']
    model_name = m['model_name']
    evaluations = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('csv'):
                file_path = os.path.join(dirpath, filename)
                print(f'Processing file: {file_path}')
                results = pd.read_csv(file_path, sep='\t', encoding="utf8")
                print(f"original size: {len(results)}")
                merged_df = pd.merge(reference_df, results, on='filename', how='outer')
                filtered_results = merged_df.dropna(subset=['result', 'label'])[:200]
                print(f"after filtering size: {len(filtered_results)}")

                bertscore = bert_score(filtered_results['label'], filtered_results['result'], model, tokenizer)
                evaluations.append({'filename': filename, 'bertscore': bertscore})

    save_csv_file(os.path.join('results','test',f"evaluations_bertscore_{today_format}_{model_name}.csv"), evaluations)
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

scorer = BERTScorer(model_type='bert-base-uncased')

def bert_score3(references, candidates, model, tokenizer):
    # BERTScore calculation
    evaluations = []
    for (reference, candidate) in zip(references, candidates):
        P, R, F1 = scorer.score([candidate], [reference])
        print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
        evaluations.append(F1.mean().item())
    if len(evaluations) == 0:
        return 0
    return statistics.mean(evaluations)


def bert_score2(references, candidates, model, tokenizer):
    evaluations = []
    for (reference, candidate) in zip(references, candidates):
        # Step 4: Prepare the texts for BERT
        inputs1 = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)

        # Step 5: Feed the texts to the BERT model
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # Step 6: Obtain the representation vectors
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

        # Step 7: Calculate cosine similarity
        similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        # Step 8: Print the result
        evaluations.append(similarity[0][0])
    if len(evaluations) == 0:
        return 0
    return statistics.mean(evaluations)


def bert_score1(references, candidates):
    # Ensure references and candidates are lists of strings
    references = references.tolist()
    candidates = candidates.tolist()

    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Tokenize and encode the references and candidates
    encoded_references = tokenizer(references, padding=True, truncation=True, return_tensors='pt')
    encoded_candidates = tokenizer(candidates, padding=True, truncation=True, return_tensors='pt')

    # Ensure the tensors have compatible dimensions
    max_len = max(encoded_references['input_ids'].size(1), encoded_candidates['input_ids'].size(1))
    encoded_references['input_ids'] = torch.nn.functional.pad(encoded_references['input_ids'], (0, max_len - encoded_references['input_ids'].size(1)))
    encoded_candidates['input_ids'] = torch.nn.functional.pad(encoded_candidates['input_ids'], (0, max_len - encoded_candidates['input_ids'].size(1)))

    # BERTScore calculation
    with torch.no_grad():
        ref_outputs = model(**encoded_references)
        cand_outputs = model(**encoded_candidates)

    # Calculate precision, recall, and F1 score
    P = torch.mean(ref_outputs.last_hidden_state, dim=1)
    R = torch.mean(cand_outputs.last_hidden_state, dim=1)
    F1 = 2 * (P * R) / (P + R + 1e-8)
    return {'precision': P.mean().item(), 'recall': R.mean().item(), 'F1': F1.mean().item()}


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

                # bertscore1 = bert_score1(filtered_results['label'], filtered_results['result'])
                # evaluations.append({'filename': filename, 'precision': precision, 'recall': recall, 'f1': f1})
                # bertscore2 = bert_score2(filtered_results['label'], filtered_results['result'], model, tokenizer)
                bertscore3 = bert_score3(filtered_results['label'], filtered_results['result'], model, tokenizer)
                evaluations.append({'filename': filename, 'bertscore3': bertscore3})

    save_csv_file(os.path.join('results','test',f"evaluations_bertscore3_{today_format}_{model_name}.csv"), evaluations)
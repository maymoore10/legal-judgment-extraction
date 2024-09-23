from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.utils import *

# Load pre-trained sentence transformer model
models = [
    {'model_name': 'alephbert-score', 'model': SentenceTransformer('imvladikon/sentence-transformers-alephbert')},
    {'model_name': 'mbert-score', 'model': SentenceTransformer('google-bert/bert-base-multilingual-cased')},
    {'model_name': 't5base-score', 'model': SentenceTransformer('sentence-transformers/sentence-t5-base')},
    {'model_name': 'hebert-score', 'model': SentenceTransformer('avichr/heBERT_sentiment_analysis')},
    {'model_name': 'mt5-score', 'model': SentenceTransformer('google/mt5-small')},]


def semantic_similarity(sentences1, sentences2, model):
    scores = []
    # Step 1: Convert sentences to embeddings
    for sentence1, sentence2 in zip(sentences1, sentences2):
        try:
            embeddings = model.encode([prepare_input(sentence1), prepare_input(sentence2)])
        except Exception as e:
            print(f"Error: {sentence1}, {sentence2}")
            continue
        # Step 2: Compute cosine similarity between the two sentence embeddings
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        scores.append(similarity_score)

    if len(scores) == 0:
        return 0
    return statistics.mean(scores)


reference_path = annotated_sentences_test
reference_df = pd.read_csv(reference_path, sep='\t', encoding="utf8")[['filename', 'label']]
# reference_texts = reference_df['label'].astype(str).apply(prepare_input)

# path = os.path.join(results_path, 'results/test')
path = '/home/morm/legal-IR/results/final/test'
for model in models:
    model_name = model['model_name']
    model = model['model']
    evaluations = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('csv'):
                file_path = os.path.join(dirpath, filename)
                print(f'Processing file: {file_path}')
                with open(file_path, 'r') as file:
                    results = pd.read_csv(file_path, sep='\t', encoding="utf8")
                    print(f"original size: {len(results)}")
                    merged_df = pd.merge(reference_df, results, on='filename', how='outer')
                    filtered_results = merged_df.dropna(subset=['result'])
                    filtered_results = filtered_results.dropna(subset=['label'])
                    print(f"after filtering size: {len(results)}")

                    # filtered_results = merged_df.dropna(how='any')
                    similarity = semantic_similarity(filtered_results['label'], filtered_results['result'], model)
                    evaluations.append({"filename": filename, "similarity": similarity})

                    # evaluations.append(evaluate_ir_results(filename, merged_df))
                    # print(f"filename: {filename}, results: \n {predictions}")
    save_csv_file(os.path.join('results','test',f"evaluations_similarity_{today_format}_{model_name}.csv"), evaluations)




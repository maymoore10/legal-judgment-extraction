import os.path

import pandas as pd
import csv
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import statistics
from utils.utils import *
# nltk.download('punkt')


def tokenize(sentences):
    """Tokenize a list of sentences."""
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def precision(retrieved, relevant):
    """Calculate precision."""
    true_positives = len(set(retrieved) & set(relevant))
    total_retrieved = len(retrieved)
    return true_positives / total_retrieved if total_retrieved > 0 else 0


def recall(retrieved, relevant):
    """Calculate recall."""
    true_positives = len(set(retrieved) & set(relevant))
    total_relevant = len(relevant)
    return true_positives / total_relevant if total_relevant > 0 else 0


def f1_score(precision_score, recall_score):
    """Calculate F1 Score."""
    if precision_score + recall_score == 0:
        return 0
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)


def accuracy(results):
    accuracies = []
    filtered_results = results.dropna(how='any')
    texts = filtered_results.astype(str).tolist()
    for text in texts:
        top_sentences, sorted_sentences, max_score, avg_score = rate_sentences(text.replace('.',  ' '))
        if max_score >= 10:
            accuracies.append(1)
        elif 7 <= max_score < 10:
            accuracies.append(0.5)
        else:
            accuracies.append(0)
    if len(accuracies) == 0:
        return 0
    return statistics.mean(accuracies)

def bleu_score(retrieved, relevant):
    scores = []
    for candidate, reference in zip(retrieved, relevant):
        """Calculate BLEU score."""
        # tokenized_candidate = ' '.join(tokenize(candidate))
        # tokenized_reference = ' '.join(tokenize(reference))

        smoothing_function = SmoothingFunction().method1
        scores.append(sentence_bleu([reference], candidate, smoothing_function=smoothing_function))
    if len(scores) == 0:
        return 0
    return statistics.mean(scores)

def rouge_score(retrieved, relevant):
    """Calculate ROUGE score."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(relevant), ' '.join(retrieved))
    return scores


def tokenize_tlnls(text):
    return text.split()


def normalized_levenshtein_similarity(token1, token2):
    distance = Levenshtein.distance(token1, token2)
    max_length = max(len(token1), len(token2))
    return 1 - (distance / max_length)  # Similarity score


def calc_tlnls_score(predicted_span, gold_span):
    P_tokens = tokenize_tlnls(predicted_span)
    G_tokens = tokenize_tlnls(gold_span)

    score = 0
    for g_token in G_tokens:
        max_similarity = max(normalized_levenshtein_similarity(g_token, p_token) for p_token in P_tokens)
        score += max_similarity

    return score / max(len(G_tokens), len(P_tokens))

def tlnls(retrieved, relevant):
    scores = []
    for sentence, reference in zip(retrieved, relevant):
        scores.append(calc_tlnls_score(sentence, reference))

    if len(scores) > 0:
        return statistics.mean(scores)
    else:
        return 0


def evaluate_ir_results(filename, texts):
    """Evaluate information retrieval results using multiple metrics."""
    # Tokenize the retrieved and relevant verdicts
    # relevant = texts['rule_based_sentences_no_prep'].astype(str)
    filtered_results = texts.dropna(subset=['result'])
    print(f'filtered: {len(filtered_results)}')
    evaluations = []

    relevant = filtered_results['label'].astype(str).apply(prepare_input)

    # tokenized_relevant = relevant
    tokenized_relevant = [' '.join(tokens) for tokens in tokenize(relevant)]

    for column in filtered_results.columns:
        if column == 'result':
            retrieved = filtered_results[column].astype(str).apply(prepare_input)

            # retrieved = retrieved.apply(prepare_input)

            # tokenized_retrieved = retrieved
            tokenized_retrieved = [' '.join(tokens) for tokens in tokenize(retrieved)]

            # tokenized_retrieved_eng = translate_to_heb_eng(retrieved)

            # precision_score = precision(tokenized_retrieved, tokenized_relevant)
            # recall_score = recall(tokenized_retrieved, tokenized_relevant)
            # f1 = f1_score(precision_score, recall_score)
            bleu = bleu_score(tokenized_retrieved, tokenized_relevant)
            acc = accuracy(retrieved)
            # rouge = rouge_score(tokenized_retrieved, tokenized_relevant)
            # rouge = 0
            tlnls_score = tlnls(relevant, retrieved)
    return {"filename": filename, "accuracy": acc, "bleu": bleu, 'tlnls': tlnls_score}
    # return {"filename": filename, "precision": precision_score,
    #                     "recall": recall_score, "f1": f1,
    #                     "accuracy": acc, "bleu": bleu, "rouge": rouge, 'tlnls': tlnls_score}


reference_path = annotated_sentences_test
reference_df = pd.read_csv(reference_path, sep='\t', encoding="utf8")[['filename', 'label']]
# reference_texts = reference_df['label'].astype(str).apply(prepare_input)

# path = os.path.join(results_path, 'results/test')
path = '/home/morm/legal-IR/results/final/test'
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
                # filtered_results = merged_df.dropna(how='any')
                evaluations.append(evaluate_ir_results(filename, merged_df))
                # print(f"filename: {filename}, results: \n {predictions}")
save_csv_file(os.path.join('results','test', f"evaluations_{today_format}.csv"), evaluations)




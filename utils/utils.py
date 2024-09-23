import csv
import json
import os
import random
import re
import statistics

import chardet
import pandas as pd
import torch
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
import numpy as np

# Download NLTK Hebrew stopwords (you might need to download these if not already present)
# nltk.download('punkt')
# nltk.download('stopwords')
hebrew_stopwords = set(stopwords.words('hebrew'))

mixed_dir = '/home/morm/legal-IR/resources/cases/mixed'
test_dir = '/home/morm/legal-IR/resources/cases/test'
criminal_dir = '/home/morm/legal-IR/resources/cases/criminal'
models_path = '/home/morm/legal-IR/models'
annotated_sentences_criminal = '/home/morm/legal-IR/resources/annotated_sentences_criminal.csv'
annotated_sentences_mixed = '/home/morm/legal-IR/resources/annotated_sentences_mixed.csv'
annotated_sentences_test = '/home/morm/legal-IR/resources/annotated_sentences_test.csv'
annotated_json = '/home/morm/legal-IR/resources/annotated_json.json'
none_sentences = '/home/morm/legal-IR/resources/none_sentence.csv'
results_path = '/home/morm/legal-IR/results'
resources_path = '/home/morm/legal-IR/resources'

keywords = {
    'תתייצב': 7,
    'יתייצב': 7,
    'סוף דבר:': 8,
    'סופ דבר:': 8,
    'העתירה נדחתה אפוא': 12,
    'כנס למעצר': 7,
    'הערעור נדחה אפוא': 12,
    'העותר יישא': 10,
    'העתירה נדחית': 10,
    'ראינו לקבוע': 10,
    'חייב בית הדין': 10,
    'החלטנו על': 7,
    'החלטנו אפוא': 7,
    'הוחלט': 7,
    'מצטרפ': -5,
    'אם תישמע דעתי': -10,
    'אמ תישמע דעתי': -10,
    'אציע לחבריי': -10,
    'פ ס ק - ד י ן': 10,
    'אנו מחליטים': 10,
    'אנו מחליטימ': 10,
    'הוסכם כי': 10,
    'אנו דוחים': 10,
    'אנו מקבלים': 10,
    'דין הערעור': 10,
    'דין העתירה': 10,
    'העותרת תשלם': 10,
    'המערערת תישא': 10,
    'המערער ישא': 10,
    'העותר ישלם': 10,
    'המערערים ישלמו': 10,
    'התוצאה היא כי': 10,
    'הוחלט ברוב דעות כאמור בפסק': -10,
    'הוחלט כאמור בפסק': -10,
    'ש ו פ ט ת': -10,
    'ש ו פ ט': -10,
    'הוחלט כאמור בפסק': -10,

}

remove_words = ['ש ו פ ט ת', 'ש ו פ ט', 'אשר על כן -', 'אשר על כן ']


def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().rstrip()
    except FileNotFoundError:
        return "NONE"


def read_file_with_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read the file with the detected encoding
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()


def rename_file(file_path):
    if file_path.endswith('.doc'):
        # Create the new file path with .docx extension
        new_file_path = file_path[:-4] + '.docx'

        # Rename the file
        os.rename(file_path, new_file_path)


def save_csv_file(path, df):
    with open(path, "w") as f:
        wr = csv.DictWriter(f, delimiter="\t", fieldnames=list(df[0].keys()))
        wr.writeheader()
        wr.writerows(df)


def score_sentence(sentence):
    score = 0
    for keyword, value in keywords.items():
        if re.search(re.escape(keyword), sentence):
            score += value
    return score


def get_last_n_sentences(text, n=1):
    # Split text into criminal
    sentences = re.split(r'(?<=[.!?]) +', text)
    # Return the last sentence
    num_sentences = min(len(sentences), n)
    return ". ".join(sentences[-num_sentences:]) if sentences else ""


def rate_sentences(text):
    if text is None:
        return "none", "none", 0, 0
    sentences = text.split(". ")

    # Score criminal and store results in a dictionary
    sentence_scores = {sentence: score_sentence(sentence) for sentence in sentences}

    # Find the sentence(s) with the highest score

    highest_scoring_sentences = [sentence for sentence, score in sentence_scores.items() if score >= 3]
    max_score = max(sentence_scores.values())
    try:
        avg_score = statistics.mean([score for sentence, score in sentence_scores.items() if score >= 3])
    except statistics.StatisticsError:
        avg_score = 0
    sorted_sentences = sorted(highest_scoring_sentences, key=lambda item: item[1], reverse=True)
    if len(sorted_sentences) == 1:
        top_sentences_str = sorted_sentences[0]
    elif len(sorted_sentences) >= 1:
        top_sentences_str = ". ".join(sorted_sentences[0:1])
    else:
        top_sentences_str = "none"
    return top_sentences_str, sorted_sentences, max_score, avg_score


def format_date(date):
    # Format the date and time as a string for a file name
    return date.strftime("%Y%m%d")


today_format = format_date(datetime.now())


def remove_sentences_with_phrases(text):
    # Split text into sentences
    phrases = ["ניתן היום", "מרכז מידע", "ש ו פ", 'ה נ ש', '']
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Filter out sentences that contain any of the phrases
    filtered_sentences = [
        sentence for sentence in sentences if not any(phrase in sentence for phrase in phrases)
    ]

    # Join the remaining sentences back into a single string
    return ". ".join(filtered_sentences)


def clean_text(text):
    # Remove punctuation and unwanted characters
    # text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Create a regex pattern to match either of the phrases
    # text = remove_sentences_with_phrases(text)
    return text


def normalize_text(text):
    # Normalize final letters (optional based on task)
    final_letters_map = {
        'ך': 'כ',
        'ם': 'מ',
        'ן': 'נ',
        'ף': 'פ',
        'ץ': 'צ'
    }
    for final_letter, regular_letter in final_letters_map.items():
        text = text.replace(final_letter, regular_letter)
    return text


def replace_final_letters(text):
    # Normalize final letters (optional based on task)
    final_letters_map = {
        'כ ': 'ך ',
        'מ ': 'ם ',
        'נ ': 'ן ',
        'פ ': 'ף ',
        'צ ': 'ץ '
    }
    for final_letter, regular_letter in final_letters_map.items():
        text = text.replace(final_letter, regular_letter)
    return text


def tokenize_text(text):
    tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
    tokens = tokenizer.tokenize(text)
    return tokens


def remove_stopwords(tokens):
    tokens = [word for word in tokens if word not in hebrew_stopwords]
    return tokens


def text_to_vector(text):
    tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
    inputs = tokenizer(text, return_tensors='pt')
    return inputs


def prepare_input(text):
    if text is None:
        return ""
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    # tokenized_text = tokenize_text(normalized_text)
    # filtered_tokens = remove_stopwords(tokenized_text)
    #vectorized_text = text_to_vector(normalized_text)
    return normalized_text
    # print(f"Original Text: {text}")
    # print(f"Cleaned Text: {cleaned_text}")
    # print(f"Normalized Text: {normalized_text}")
    # print(f"Tokenized Text: {tokenized_text}")
    # print(f"Tokens after Stopword Removal: {filtered_tokens}")
    # print(f"Vectorized Text: {vectorized_text}")

def contains_hebrew(text):
    # Regular expression pattern to match Hebrew characters
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    # Search for Hebrew characters in the text
    try:
        return bool(hebrew_pattern.search(text))
    except:
        return False


def remove_redundant_sentences(text):
    patterns = ['www', 'txt', 'ש ו פ ט', 'ניתנה היו', 'ניתן היו', 'העתק נאמן', 'מרכז מידע']
    regex = re.compile(r'[A-Za-z]\d+')
    #internal_patterns = ['']
    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered_sentences = [
        sentence for sentence in sentences if not any(pattern in sentence for pattern in patterns)
                                              and not regex.search(sentence)
                                              and len(sentence) > 5
    ]
    return ". ".join(filtered_sentences)


def cluster_sentences(sentences, embeddings, num_clusters=5):
    num_clusters = min(np.size(embeddings, 0), num_clusters)
    # print(f"senteces: {criminal}, embeddings: {embeddings}. numclusters: {num_clusters}")
    # Cluster criminal
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)

    # print(kmeans.get_params())
    try:
        clustered_sentences = {i: [] for i in range(kmeans.n_clusters)}
    except:
        clustered_sentences = {i: [] for i in range(2)}

    for sentence_id, cluster_id in enumerate(kmeans.labels_):
        clustered_sentences[cluster_id].append(sentences[sentence_id])
    return clustered_sentences


def summarize_verdicts(clustered_sentences):
    # Assume the most relevant cluster is the largest one
    largest_cluster = max(clustered_sentences, key=lambda k: len(clustered_sentences[k]))
    summary = " ".join(clustered_sentences[largest_cluster])
    return remove_redundant_sentences(summary)


def segment_and_embed(model, text):
    # Split text into criminal
    sentences = re.split(r'(?<=[.!?]) +', text)
    # criminal = [preprocess_text(sentence) for sentence in criminal if sentence]
    # Compute embeddings
    embeddings = model.encode(sentences, show_progress_bar=True)
    return sentences, embeddings


questions = [
    '?מה גזר הדין בתיק זה?',
    'מה הוחלט בתיק זה?',
    'מה ההכרעה בתיק זה?'
]


def load_data(path=annotated_sentences_mixed, files_dir=mixed_dir, question='מה גזר הדין בתיק זה?'):
    data = pd.read_csv(path, sep='\t', encoding="utf8")
    # data = data.drop_duplicates(subset='label')
    data['context'] = ''
    for idx, row in data.iterrows():
        file_path = os.path.join(files_dir, row['fileName'])
        label = row['label']
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'r', encoding="utf8") as file:
                context = file.read().strip()
                answer_start = context.find(label.split(". ")[0])

                data.at[idx, 'question'] = question
                data.at[idx, 'answer_start'] = answer_start
                data.at[idx, 'answer_text'] = label
                data.at[idx, 'context'] = prepare_input(context)
        except FileNotFoundError:
            continue
            # data.at[idx, 'context'] = 'none'  # Or you can use an empty string or another placeholder

    return data


def get_balanced_data():
    sentencing_df = pd.read_csv(annotated_sentences_mixed, sep='\t', encoding="utf8")
    non_sentencing_df = pd.read_csv(none_sentences, sep='\t', encoding="utf8")

    # Drop duplicates and get texts and labels
    sentencing_texts = sentencing_df.drop_duplicates(subset='label')['label'].tolist()
    non_sentencing_texts = non_sentencing_df.drop_duplicates(subset='label')['label'].tolist()

    # Create labels
    labels = [1] * len(sentencing_texts)
    non_sentencing_labels = [0] * len(non_sentencing_texts)

    # Combine texts and labels
    texts = sentencing_texts + non_sentencing_texts
    texts = [prepare_input(text) for text in texts]
    labels = labels + non_sentencing_labels

    return texts, labels

# model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
# heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(model_name)
# heb_to_eng_model = MarianMTModel.from_pretrained(model_name)
#
#
# def translate_to_heb_eng(texts):
#     translated_texts = []
#
#     # Translate text
#     for source_text in texts:
#         translated = heb_to_eng_model.generate(**heb_to_eng_tokenizer.prepare_seq2seq_batch([source_text], return_tensors='pt'))
#         translated_texts.append(heb_to_eng_tokenizer.decode(translated[0], skip_special_tokens=True))
#
#     return translated_texts

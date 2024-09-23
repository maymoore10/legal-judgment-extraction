from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, MT5Tokenizer, \
    MT5ForSequenceClassification
from utils.utils import *

# Load the tokenizer and model
legal_hebert_path = os.path.join(models_path, '20240911/mixed/IR_finetuned_legalhebert_learningrate5e05_4epochs_benchmark')
aleph_bert_path = os.path.join(models_path, '20240912/mixed/IR_finetuned_alephbert_learningrate5e05_4epochs_benchmark')
mt5_small_path = os.path.join(models_path, '/home/morm/legal-IR/models/20240922/mixed/IR_finetuned_mt5_learningrate1e03_4epochs')
mbert_path = os.path.join(models_path, 'results/mixed/IR_finetuned_mbert_learningrate5e05_4epochs')


models = [
    # {
    #     'model_name': 'legal-hebert-bm',
    #     'tokenizer': BertTokenizer.from_pretrained('avichr/Legal-heBERT'),
    #     'model': BertForSequenceClassification.from_pretrained('avichr/Legal-heBERT')
    # },
    # {
    #     'model_name': 'aleph-bert-bm',
    #     'tokenizer': AutoTokenizer.from_pretrained("onlplab/alephbert-base"),
    #     'model': AutoModelForSequenceClassification.from_pretrained("onlplab/alephbert-base")
    # },
    # {
    #     'model_name': 'mt5-small-bm',
    #     'tokenizer': MT5Tokenizer.from_pretrained("google/mt5-small"),
    #     'model': MT5ForSequenceClassification.from_pretrained("google/mt5-small")
    # },
    {
        'model_name': 'mbert-bm',
        'tokenizer': BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased"),
        'model': BertForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased")
    },
    # {
    #     'model_name': 'legal-hebert-ft',
    #     'tokenizer': BertTokenizer.from_pretrained(legal_hebert_path),
    #     'model': BertForSequenceClassification.from_pretrained(legal_hebert_path)
    # },
    # {
    #     'model_name': 'aleph-bert-ft',
    #     'tokenizer': AutoTokenizer.from_pretrained(aleph_bert_path),
    #     'model': AutoModelForSequenceClassification.from_pretrained(aleph_bert_path)
    # },
    # {
    #     'model_name': 'mt5-small-ft-1e03',
    #     'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path),
    #     'model': MT5ForSequenceClassification.from_pretrained(mt5_small_path)
    # },
    {
        'model_name': 'mbert-ft',
        'tokenizer': BertTokenizer.from_pretrained(mbert_path),
        'model': BertForSequenceClassification.from_pretrained(mbert_path)
    }
]


# Function to classify criminal
def classify_and_filter_sentences(model, tokenizer, text):
    # Split the text into criminal
    sentences = text.split('. ')
    # Prepare a list to store the results
    filtered_sentences = []

    for sentence in sentences:
        txt = nltk.sent_tokenize(prepare_input(sentence))
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get model outputs (logits)
        outputs = model(**inputs)
        logits = outputs.logits

        # Use argmax to get the index of the class with the highest score (0 or 1)
        predicted_label = torch.argmax(logits, dim=1).item()

        # Filter out criminal where the prediction is 0
        if len(txt) > 0 and predicted_label == 1:
            filtered_sentences.append(sentence)

    return filtered_sentences


path = criminal_dir

for model_obj in models:
    answers = []
    model_name = model_obj['model_name']
    model = model_obj['model']
    tokenizer = model_obj['tokenizer']
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        # Check if it is a file
        with open(file_path, 'r') as file:
            text = prepare_input(file.read().rstrip())
            sentence_classifications = classify_and_filter_sentences(model, tokenizer, text)
            cleaned_text = clean_text('. '.join(sentence_classifications))
            answers.append({'filename': filename, 'result': cleaned_text})
        if len(answers) > 200:
            break
    save_csv_file(os.path.join(results_path, f"{today_format}/criminal/information_retrieval_{model_name}.csv"), answers)


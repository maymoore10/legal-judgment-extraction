import random

from transformers import AutoModelForQuestionAnswering, MT5ForQuestionAnswering, BertForQuestionAnswering, \
    BertTokenizer, MT5Tokenizer
from utils.utils import *


legal_hebert_path = os.path.join(models_path, '/home/morm/legal-IR/models/20240918/mixed/finetuned_legalhebert_qa_3e-05_q0')
aleph_bert_path = os.path.join(models_path, '/home/morm/legal-IR/models/20240918/mixed/finetuned_alephbert_qa_3e-05_q0')
mt5_small_path_q0 = os.path.join(models_path, '/home/morm/legal-IR/models/20240920/mixed/finetuned_mt5_small_qa_1e-3_q0')
mt5_small_path_q1 = os.path.join(models_path, '/home/morm/legal-IR/models/20240921/mixed/finetuned_mt5_small_qa_3e-5_q1')
mt5_small_path_q2 = os.path.join(models_path, '/home/morm/legal-IR/models/20240921/mixed/finetuned_mt5_small_qa_3e-5_q2')
mbert_path = os.path.join(models_path, '/home/morm/legal-IR/models/20240923/mixed/finetuned_mbert_qa_3e-5_q0')

models = [
    {
        'model_name': 'mbert-bm',
        'tokenizer': BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased'),
        'model': BertForQuestionAnswering.from_pretrained('google-bert/bert-base-multilingual-cased'),
        'question': questions[0]

    },
    {
        'model_name': 'mbert-ft',
        'tokenizer': BertTokenizer.from_pretrained(mbert_path),
        'model': BertForQuestionAnswering.from_pretrained(mbert_path),
        'question': questions[0]

    },
    {
        'model_name': 'legal-hebert-bm',
        'tokenizer': BertTokenizer.from_pretrained('avichr/Legal-heBERT'),
        'model': BertForQuestionAnswering.from_pretrained('avichr/Legal-heBERT'),
        'question': questions[0]

    },
    {
        'model_name': 'aleph-bert-bm',
        'tokenizer': AutoTokenizer.from_pretrained("onlplab/alephbert-base"),
        'model': BertForQuestionAnswering.from_pretrained("onlplab/alephbert-base"),
        'question': questions[0]

    },
    {
        'model_name': 'mt5-small-bm',
        'tokenizer': MT5Tokenizer.from_pretrained("google/mt5-small"),
        'model': MT5ForQuestionAnswering.from_pretrained("google/mt5-small"),
        'question': questions[0]

    },
    {
        'model_name': 'legal-hebert-ft',
        'tokenizer': BertTokenizer.from_pretrained(legal_hebert_path),
        'model': BertForQuestionAnswering.from_pretrained(legal_hebert_path),
        'question': questions[0]
    },
    {
        'model_name': 'aleph-bert-ft',
        'tokenizer': AutoTokenizer.from_pretrained(aleph_bert_path),
        'model': BertForQuestionAnswering.from_pretrained(aleph_bert_path),
        'question': questions[0]
    },
    # {
    #     'model_name': 'mt5-small-ft-3e5-3q',
    #     'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path_3e5_3q),
    #     'model': MT5ForQuestionAnswering.from_pretrained(mt5_small_path_3e5_3q),
    #     'question': None
    # },
    # {
    #     'model_name': 'mt5-small-ft-1e3-3q',
    #     'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path_1e3_3q),
    #     'model': MT5ForQuestionAnswering.from_pretrained(mt5_small_path_1e3_3q),
    #     'question': None
    # },
    {
        'model_name': 'mt5-small-ft-q0',
        'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path_q0),
        'model': MT5ForQuestionAnswering.from_pretrained(mt5_small_path_q0),
        'question': questions[0]
    },
    {
        'model_name': 'mt5-small-ft-q1',
        'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path_q1),
        'model': MT5ForQuestionAnswering.from_pretrained(mt5_small_path_q1),
        'question': questions[2]

    },
    # {
    #     'model_name': 'mt5-small-ft-q2',
    #     'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path_q2),
    #     'model': MT5ForQuestionAnswering.from_pretrained(mt5_small_path_q2),
    #     'question': questions[2]
    #
    # }

]

file_names = []
texts = []
path = criminal_dir
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    # Check if it is a file
    with open(file_path, 'r') as file:
        text = file.read().rstrip()
        texts.append(text)
        file_names.append(filename)


# Adjusting tokenization to handle both question and context without exceeding the limit
# questions = ["מהי הפסיקה המשפטית?"] * len(texts)
# questions = ["מה ההחלטה המשפטית?"] * len(texts)
# questions = ["what is the verdict decision?"] * len(texts)
max_length = 512  # BERT's maximum input length

for model_obj in models:
    answers = []
    input_ids = []
    attention_masks = []
    model_name = model_obj['model_name']
    model = model_obj['model']
    tokenizer = model_obj['tokenizer']
    question = model_obj['question']
    if question is None:
        questions = random.choices(questions, k=len(texts))
    else:
        questions = [question] * len(texts)

    inputs = tokenizer(questions, texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    for idx, text in enumerate(texts):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,
            truncation=True,  # Ensure sequences are truncated to `max_length`
            padding='max_length',  # Pad to `max_length`
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        model.eval()
        with torch.no_grad():
            outputs = model(**encoded)

            answer_start_index = torch.argmax(outputs.start_logits)
            answer_end_index = torch.argmax(outputs.end_logits)

            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            answers.append({"filename": file_names[idx], "result": tokenizer.decode(predict_answer_tokens)})

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    save_csv_file(os.path.join(results_path, f"{today_format}/criminal/question_answering_{model_name}_3e5.csv"), answers)




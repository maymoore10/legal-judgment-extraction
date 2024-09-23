import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BertTokenizerFast, BertForQuestionAnswering
from utils.utils import *

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
model = BertForQuestionAnswering.from_pretrained('onlplab/alephbert-base')


class QADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def preprocess_function(row):
    question = row['question']
    context = row['context']
    answer = row['label']

    # Tokenize question and context
    inputs = tokenizer(question, context, max_length=512, truncation=True, padding='max_length', return_tensors='pt',
                       return_offsets_mapping=True)

    # Find the start and end positions of the answer in the context
    start_pos = context.find(answer)
    end_pos = start_pos + len(answer)

    offset_mapping = inputs['offset_mapping'][0]

    start_token_idx = end_token_idx = None
    for idx, (start, end) in enumerate(offset_mapping):
        if start == start_pos:
            start_token_idx = idx
        if end == end_pos:
            end_token_idx = idx
            break

    # If we didn't find the answer in the context, skip this example
    if start_token_idx is None or end_token_idx is None:
        return None

    # Remove offset mapping as it is not needed for training
    inputs.pop("offset_mapping")

    inputs['start_positions'] = torch.tensor(start_token_idx)
    inputs['end_positions'] = torch.tensor(end_token_idx)

    return {key: val.squeeze() for key, val in inputs.items()}


# Load and preprocess the data
df = load_data(question=questions[0])

# Process the data and filter out None values
processed_data = df.apply(preprocess_function, axis=1).dropna()
processed_data = [data for data in processed_data if data is not None]  # Filter out None values

# Extract input_ids, attention_masks, start_positions, and end_positions for training
input_ids = [entry['input_ids'] for entry in processed_data]
attention_mask = [entry['attention_mask'] for entry in processed_data]
start_positions = [entry['start_positions'] for entry in processed_data]
end_positions = [entry['end_positions'] for entry in processed_data]

# Split the data into training and validation sets
train_data, val_data = train_test_split(range(len(processed_data)), test_size=0.2, random_state=42)
train_encodings = {
    'input_ids': [input_ids[i] for i in train_data],
    'attention_mask': [attention_mask[i] for i in train_data],
    'start_positions': [start_positions[i] for i in train_data],
    'end_positions': [end_positions[i] for i in train_data]
}
val_encodings = {
    'input_ids': [input_ids[i] for i in val_data],
    'attention_mask': [attention_mask[i] for i in val_data],
    'start_positions': [start_positions[i] for i in val_data],
    'end_positions': [end_positions[i] for i in val_data]
}

# Create the datasets
train_dataset = QADataset(train_encodings)
val_dataset = QADataset(val_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer  # Pass the tokenizer to the trainer
)

# Fine-tune the model
trainer.train()

output_dir = os.path.join(models_path, f'{today_format}/mixed/finetuned_alephbert_qa_3e-05_q0')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

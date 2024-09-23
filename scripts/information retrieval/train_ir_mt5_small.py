import os
import torch
from sklearn.model_selection import train_test_split
from transformers import MT5Tokenizer, MT5ForSequenceClassification, Trainer, TrainingArguments
import pandas as pd  # Assuming you are using pandas for reading CSV
from utils.utils import *  # Ensure utils are properly imported


# Dataset class
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Load the data

# Get balanced data
texts, labels = get_balanced_data()

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.33, random_state=42)

# Load a pre-trained BERT tokenizer
model_name = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create dataset objects
train_dataset = LegalDataset(train_encodings, train_labels)
test_dataset = LegalDataset(test_encodings, test_labels)

# Load the pre-trained BERT model
model = MT5ForSequenceClassification.from_pretrained(model_name)

# Set training arguments
num_epochs = 4
learning_rate = 5e03
training_args = TrainingArguments(
    output_dir='../tfidf/train_mt5/results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=100,  # Consider lowering this if dataset is small
    weight_decay=0.01,
    learning_rate=learning_rate,
    logging_dir='../tfidf/train_mt5/logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Test Accuracy: {eval_result}")

# Save the model and tokenizer
output_dir = os.path.join(models_path, f'{today_format}/mixed/IR_finetuned_mt5_learningrate5e03_4epochs')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

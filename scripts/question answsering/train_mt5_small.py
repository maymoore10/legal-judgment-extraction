from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from torch.utils.data import Dataset
import os
from utils.utils import *
from datetime import datetime

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
model = T5ForConditionalGeneration.from_pretrained('google/mt5-small')


class QADataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }


def preprocess_function(row):
    # question = row['question']
    question = 'מה גזר הדין בתיק זה?'
    context = row['context']
    answer = row['label']
    # Create the input for mT5
    inputs = "question: " + question + " context: " + context
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    # Tokenize the answer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(answer, max_length=32, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Load and preprocess the data
df = load_data(question=questions[0])

processed_data = df.apply(preprocess_function, axis=1).tolist()
# Load and preprocess the validation data
processed_train, processed_val = train_test_split(processed_data, test_size=0.2, random_state=42)

input_ids = [entry['input_ids'] for entry in processed_train]
attention_mask = [entry['attention_mask'] for entry in processed_train]
labels = [entry['labels'] for entry in processed_train]

# Convert processed validation data to lists for evaluation
input_ids_val = [entry['input_ids'] for entry in processed_val]
attention_mask_val = [entry['attention_mask'] for entry in processed_val]
labels_val = [entry['labels'] for entry in processed_val]

# Create the validation dataset
train_dataset = QADataset(input_ids, attention_mask, labels)
val_dataset = QADataset(input_ids_val, attention_mask_val, labels_val)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Pass the validation dataset here
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save model and tokenizer

output_dir = os.path.join(models_path, f'{today_format}/mixed/finetuned_mt5_small_qa_3e-5_q0')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Legal Judgment Extraction Project

This repository contains the scripts and resources used to train and evaluate models for extracting legal judgments from Hebrew court verdicts. The project applies various Natural Language Processing (NLP) techniques, including Information Retrieval (IR), Question Answering (QA), Rule-Based Methods, and Summarization.

## Repository Structure

- **resources/**: Contains any auxiliary files, such as datasets, configuration files, and model-specific resources.
  
- **results/**: Stores output from model training, including evaluation metrics, logs, and extracted judgments.

- **scripts/**: Contains all Python scripts used for training, testing, and evaluating models.

  - **evaluation/**: Contains scripts and modules used for model evaluation across experiments.

  - **information_retrieval/**: Scripts related to training and testing Information Retrieval models.

  - **question_answering/**: Contains scripts for training and testing models for the Question Answering task. Each model is fine-tuned to answer questions related to judgments in Hebrew court texts.

  - **rulebased/**: Contains the rule-based extraction script that applies predefined heuristics to extract legal judgments from verdict texts.

  - **summarization/**: Scripts related to summarizing legal texts to extract the key verdict.
    
- **utils/**: Helper scripts or utility functions used across the different experiments (e.g., for data preprocessing, logging, or evaluation metric calculations).

## Usage Instructions

**Make sure that destination directories are created before running training and test scripts**

### Example: Training Information Retrieval Models
To train an information retrieval model, navigate to the `scripts/information_retrieval/` directory and run the appropriate training script. For example, to train AlephBERT:

```bash
python train_ir_alephbert.py
```

### Example: Test Information Retrieval Models
To train an information retrieval model, navigate to the `scripts/information_retrieval/` directory and run the appropriate training script. For example, to train AlephBERT:

```bash
python test.py
```


## Model Description

- **AlephBERT/**: A Hebrew BERT model pre-trained on general Hebrew text and fine-tuned for legal-specific tasks like information retrieval and question answering.
- **Legal-HeBERT/**: A BERT-based model specifically trained on Hebrew legal text, optimized for tasks involving legal document processing.
- **mBERT/**: Multilingual BERT model capable of processing Hebrew text, fine-tuned for information retrieval.
- **mT5-small/**: A multilingual version of the T5 model, fine-tuned for sequence-to-sequence tasks like summarization and question answering in Hebrew.


  ## Requirements

- **Python 3.8+/**
- **Huggingface Transformers/**
- **PyTorch/**
- **Scikit-learn/**
- **sentence_transformers/**
- **nltk/**
- **numpy/**
- **Pandas/**


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
The script will store the model in `models/yyyymmdd/your-path`

### Example: Test Information Retrieval Models
To evaluate a trained model, use the test.py script inside the `scripts/information_retrieval/` directory and run the appropriate test script:

```bash
python test.py
```
The script store test results in `results/yyyymmdd/your-path`

### Example: Evaluate
To evaluate results, use the evaluate.py, bertscore.py and/or semantic-similarity.py script inside the `scripts/evaluations/` directory and run the appropriate script:

```bash
python evaluate.py
```
The script will go over test results and score the perfomance based on the relavant metric and save it in `scripts/evaluations/results/yyyymmdd`

## Model Description

- **AlephBERT/**: A Hebrew BERT model pre-trained on general Hebrew text and fine-tuned for legal-specific tasks like information retrieval and question answering.
- **Legal-HeBERT/**: A BERT-based model specifically trained on Hebrew legal text, optimized for tasks involving legal document processing.
- **mBERT/**: Multilingual BERT model capable of processing Hebrew text, fine-tuned for information retrieval.
- **mT5-small/**: A multilingual version of the T5 model, fine-tuned for sequence-to-sequence tasks like summarization and question answering in Hebrew.

## Adding Model for Test Execution
If you wish to run test on multiple models, you can add to the models list a definition of the model: name, tokenizer, and model.for example:
```bash
    {
        'model_name': 'mt5-ft-3e05',
        'tokenizer': MT5Tokenizer.from_pretrained(mt5_small_path3e05),
        'model': MT5ForSequenceClassification.from_pretrained(mt5_small_path3e05)
    }
```


  ## Requirements

- **Python 3.8+**
- **Huggingface Transformers**
- **PyTorch**
- **Scikit-learn**
- **sentence_transformers**
- **nltk**
- **numpy**
- **Pandas**


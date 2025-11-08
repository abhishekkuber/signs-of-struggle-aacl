'''
Preliminary XLMR Finetuning script in Table 1 in the paper.
Train : EN
Test : EN, NL, KT
'''

CURRENT_FOLD = 0
DATASET = 'kt'

############################################################################################################################################################
## LIBRARIES
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import torch
import evaluate

# FILL IN YOUR PATHS HERE
DATA_PATH = "..."
CACHE_DIR="..."
OUTPUT_DIR="..."

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################################################################################################################################
## FUNCTIONS
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

# Custom metric function that computes accuracy, F1 score and prints classification report
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    print(classification_report(labels, predictions))
    return {**accuracy, **f1}

def encode_batch(batch):
  return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

############################################################################################################################################################
## DATA

dataset = pd.read_csv(f"{DATA_PATH}/cognitive_distortions.csv")
english_texts = dataset['Patient Question'].tolist()
english_labels = dataset['Dominant Distortion'].apply(lambda x: 0 if x == 'No Distortion' else 1).tolist()

# Create the train test splits based on the dataset type.

if DATASET == 'kt':
  data = pd.read_csv(f"{DATA_PATH}/kindertelefoon_500.csv")
  data = data.dropna(subset=['label'])
  data = data.reset_index(drop=True)
  other_texts = data['text'].tolist()
  other_labels = data['label'].apply(lambda x: 0 if x=='not distorted' else 1).tolist()

elif DATASET == 'nl':
  dataset = pd.read_csv(f"{DATA_PATH}/dutch_cognitive_distortions.csv")
  other_texts = dataset['text'].tolist()
  other_labels = dataset['distortion'].apply(lambda x: 0 if x == 'No Distortion' else 1).tolist()

if DATASET != 'en':
  english_data = pd.DataFrame({
      'text': english_texts,
      'label': english_labels
  })

  other_data = pd.DataFrame({
      'text': other_texts,
      'label': other_labels
  })

  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  fold_datasets = []

  english_splits = list(kf.split(english_data))
  other_splits = list(kf.split(other_data))

  for fold in range(5):
      print(f"Fold {fold+1}")

      english_train_idx, english_test_idx = english_splits[fold]
      dutch_train_idx, dutch_test_idx = other_splits[fold]

      english_train = english_data.iloc[english_train_idx].reset_index(drop=True)
      english_test = english_data.iloc[english_test_idx].reset_index(drop=True)
      dutch_train = other_data.iloc[dutch_train_idx].reset_index(drop=True)
      dutch_test = other_data.iloc[dutch_test_idx].reset_index(drop=True)

      train_dataset = Dataset.from_pandas(english_train)
      test_dataset = Dataset.from_pandas(dutch_test)

      dataset = DatasetDict({
          'train': train_dataset,
          'test': test_dataset
      })

      fold_datasets.append(dataset)
else:
  english_data = pd.DataFrame({
      'text': english_texts,
      'label': english_labels
  })

  kf = KFold(n_splits=5, shuffle=True, random_state=42)

  fold_datasets = []

  english_splits = list(kf.split(english_data))

  for fold in range(5):
      print(f"Fold {fold+1}")

      english_train_idx, english_test_idx = english_splits[fold]
      english_train = english_data.iloc[english_train_idx].reset_index(drop=True)
      english_test = english_data.iloc[english_test_idx].reset_index(drop=True)

      train_dataset = Dataset.from_pandas(english_train)
      test_dataset = Dataset.from_pandas(english_test)

      dataset = DatasetDict({
          'train': train_dataset,
          'test': test_dataset
      })

      fold_datasets.append(dataset)

dataset = fold_datasets[CURRENT_FOLD]


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir=CACHE_DIR)
dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


############################################################################################################################################################
## MODEL
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2, cache_dir=CACHE_DIR)
model.to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to="none",
    num_train_epochs=6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    eval_strategy="no",
    no_cuda=False, 
    local_rank=-1,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

print("\nBefore Training!")
trainer.evaluate()

print ("\nTraining!")
trainer.train()

print("\nAfter Training!")
trainer.evaluate()
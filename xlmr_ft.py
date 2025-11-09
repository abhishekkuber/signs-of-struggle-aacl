'''
Script to run XLMR Finetuning (XLMR Ft) in Table 2. 
Train Data : EN + KT
Test Data : KT
'''
CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES
from transformers import XLMRobertaForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
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

data = pd.read_csv(f"{DATA_PATH}/kindertelefoon_500.csv")
data = data.dropna(subset=['label'])
data = data.reset_index(drop=True)

dutch_texts = data['text'].tolist()
dutch_labels = data['label'].apply(lambda x: 0 if x=='not distorted' else 1).tolist()

english_data = pd.DataFrame({
    'text': english_texts,
    'label': english_labels
})

dutch_data = pd.DataFrame({
    'text': dutch_texts,
    'label': dutch_labels
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_datasets = []

english_splits = list(kf.split(english_data))
dutch_splits = list(kf.split(dutch_data))

for fold in range(5):
    english_train_idx, english_test_idx = english_splits[fold]
    dutch_train_idx, dutch_test_idx = dutch_splits[fold]

    english_train = english_data.iloc[english_train_idx].reset_index(drop=True)
    english_test = english_data.iloc[english_test_idx].reset_index(drop=True)
    dutch_train = dutch_data.iloc[dutch_train_idx].reset_index(drop=True)
    dutch_test = dutch_data.iloc[dutch_test_idx].reset_index(drop=True)

    train_data_combined = pd.concat([english_train, dutch_train], ignore_index=True)
    test_data_combined = pd.concat([english_test, dutch_test], ignore_index=True)

    train_dataset = Dataset.from_pandas(train_data_combined)
    test_dataset = Dataset.from_pandas(dutch_test)

    # Create DatasetDict for this fold
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
    output_dir=f"{OUTPUT_DIR}/xlmr_finetuning_trainer_fold_{CURRENT_FOLD}",
    report_to="none",
    num_train_epochs=6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
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


# In case you want to print the predictions, useful for doing the McNemar's test later on
predictions = trainer.predict(dataset['test'])
logits = predictions.predictions
y_pred = [int(x) for x in logits.argmax(axis=1)]
y_true = [int(x) for x in predictions.label_ids]
print(f"Predictions: {y_pred}")
print(f"True labels: {y_true}")
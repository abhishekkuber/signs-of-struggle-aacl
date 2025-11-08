'''
Script to run XLMR Finetuning for the rewritten English data in Table 2. 
Train : Rewritten EN 
Test : KT
'''
CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES

from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import evaluate
import json
import torch

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

texts = []
with open(f'{DATA_PATH}/teenage_cog_dist.json', 'r') as f:
  texts = json.load(f)

teenager_texts = []
teenager_labels = []
for text in texts:
  teenager_texts.append(text['teenage_translation'])
  teenager_labels.append(text['label'])

data = pd.read_csv(f'{DATA_PATH}/kindertelefoon_500.csv')
data = data.dropna(subset=['label'])
data = data.reset_index(drop=True)

dutch_texts = data['text'].tolist()
dutch_labels = data['label'].apply(lambda x: 0 if x=='not distorted' else 1).tolist()

teenager_data = pd.DataFrame({
    'text': teenager_texts,
    'label': teenager_labels
})

dutch_data = pd.DataFrame({
    'text': dutch_texts,
    'label': dutch_labels
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_datasets = []

english_splits = list(kf.split(teenager_data))
dutch_splits = list(kf.split(dutch_data))

for fold in range(5):
    print(f"Fold {fold+1}")

    english_train_idx, english_test_idx = english_splits[fold]
    dutch_train_idx, dutch_test_idx = dutch_splits[fold]

    english_train = teenager_data.iloc[english_train_idx].reset_index(drop=True)
    english_test = teenager_data.iloc[english_test_idx].reset_index(drop=True)
    dutch_train = dutch_data.iloc[dutch_train_idx].reset_index(drop=True)
    dutch_test = dutch_data.iloc[dutch_test_idx].reset_index(drop=True)

    train_data_combined = pd.concat([english_train, dutch_train], ignore_index=True)
    test_data_combined = pd.concat([english_test, dutch_test], ignore_index=True)

    train_dataset = Dataset.from_pandas(pd.concat([english_train, english_test], ignore_index=True))
    test_dataset = Dataset.from_pandas(dutch_test)

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
## SETTING UP THE MODEL AND TRAINING

model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2, cache_dir=CACHE_DIR)
model.to(device)

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/xlmr_rewritten_trainer_finetuning_fold_{CURRENT_FOLD}",
    eval_strategy="epoch",
    report_to="none",
    learning_rate=2e-5,                 
    per_device_train_batch_size=2,      
    per_device_eval_batch_size=2,       
    gradient_accumulation_steps=4,
    num_train_epochs=3,                 
    weight_decay=0.01,               
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
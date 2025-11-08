'''
Script to run XLMR Adapter in Table 2. 
Train : EN + KT
Test : KT
'''

CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES
import pandas as pd
import evaluate
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import TrainingArguments, XLMRobertaConfig, AutoTokenizer
from adapters import AdapterTrainer, AutoAdapterModel

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
## SETTING UP DATA 
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
## ADAPTERS
config = XLMRobertaConfig.from_pretrained("xlm-roberta-base",num_labels=2)
model = AutoAdapterModel.from_pretrained("xlm-roberta-base", config=config, cache_dir=CACHE_DIR)
model.add_adapter("binary_cog_dist_xlmr_base", config="seq_bn")
model.add_classification_head("binary_cog_dist_xlmr_base", num_labels=2, id2label={ 0: "no", 1: "yes"})
model.train_adapter("binary_cog_dist_xlmr_base")
model.to(device)

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    report_to="none",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    eval_strategy="epoch",
    output_dir=f"{OUTPUT_DIR}/adapter_trainer_fold_{CURRENT_FOLD}",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
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
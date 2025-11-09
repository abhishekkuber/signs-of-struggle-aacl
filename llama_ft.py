'''
Script to run Llama Fine Tuning (LLAMA FT) in Table 2. 
Train Data : EN + KT
Test Data : KT
'''
CURRENT_FOLD = 0
HF_TOKEN = "hf_...." # PUT IN YOUR TOKEN HERE, MAKE SURE IT HAS ACCESS TO THE META LLAMA MODELS ON HUGGINGFACE

############################################################################################################################################################
## LIBRARIES

import pandas as pd
import numpy as np
import torch
import os
import torch.nn.functional as F

from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from safetensors.torch import save_file


# FILL IN YOUR PATHS HERE
DATA_PATH = "..."
CACHE_DIR="..."
OUTPUT_DIR="..."

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "meta-llama/Llama-3.1-8B"

############################################################################################################################################################
## FUNCTIONS

def data_preprocesing(row):
    return tokenizer(row['text'], truncation=True, max_length=2048)

def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    print(classification_report(labels, predictions))
    return {'balanced_accuracy' : balanced_accuracy_score(labels, predictions),
    'accuracy':accuracy_score(labels, predictions), 'f1 score': f1_score(labels, predictions, average='weighted')}

############################################################################################################################################################
## SETTING UP THE MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer Loaded")

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

print("Model Loading")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2,
    device_map='auto',
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    offload_folder=f"{OUTPUT_DIR}/offload",
)

model.to(device)

print("Model Loaded")

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05,
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

print("Model Prepared for KBIT Training")


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
    'label': english_labels,
})

dutch_data = pd.DataFrame({
    'text': dutch_texts,
    'label': dutch_labels,
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_datasets = []

english_splits = list(kf.split(english_data))
dutch_splits = list(kf.split(dutch_data))

for fold in range(5):
    print(f"Fold {fold+1}")

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

class_weights=(1/pd.Series(dataset['train']['label']).value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
print(class_weights)


tokenized_data = dataset.map(data_preprocesing, batched=True,
remove_columns=['text'])
tokenized_data.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

############################################################################################################################################################
## CUSTOM TRAINER

# we need to create a custom Trainer class to handle the class weights and save the model's score weights during training

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.clone().detach().to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

class CustomCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        checkpoint_dir = os.path.join(output_dir, f"fold-{CURRENT_FOLD}-checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # model = kwargs.get('model')
        cls_state_dict = {'model.score': trainer.model.score.weight.detach().clone().cpu()}
        save_file(cls_state_dict, os.path.join(checkpoint_dir, 'score.safetensors'))
        return control  # Return control back to Trainer for normal operations


############################################################################################################################################################
## TRAINING THE MODEL

training_args = TrainingArguments(
    output_dir = f"{OUTPUT_DIR}/llama_cls_trainer_fold_{CURRENT_FOLD}",
    learning_rate = 1e-4,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 2,
    num_train_epochs = 1,
    weight_decay = 0.01,
    report_to="none",
    eval_strategy="epoch",
)
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data['train'],
    eval_dataset = tokenized_data['test'],
    processing_class = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
    callbacks=[CustomCheckpointCallback()]
)


print("\nBefore Training!")
trainer.evaluate()

print ("\nTraining!")
trainer.train()

print("\nAfter Training!")
trainer.evaluate()

# In case you want to print the predictions, useful for doing the McNemar's test later on
predictions = trainer.predict(tokenized_data['test'])
logits = predictions.predictions
y_pred = [int(x) for x in logits.argmax(axis=1)]
y_true = [int(x) for x in predictions.label_ids]
print(f"Predictions: {y_pred}")
print(f"True labels: {y_true}")

'''
Script to run Preliminary Llama FT, in Table 1 
Train Data : EN
Test Data : EN, NL, KT
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
en_texts = dataset['Patient Question'].tolist()
en_labels = dataset['Dominant Distortion'].apply(lambda x: 0 if x == 'No Distortion' else 1).tolist()

# Kindertelefoon data
data = pd.read_csv(f"{DATA_PATH}/kindertelefoon_500.csv")
data = data.dropna(subset=['label'])
data = data.reset_index(drop=True)
kt_texts = data['text'].tolist()
kt_labels = data['label'].apply(lambda x: 0 if x=='not distorted' else 1).tolist()

# English dataset translated to Dutch
dataset = pd.read_csv(f"{DATA_PATH}/dutch_cognitive_distortions.csv")
nl_texts = dataset['text'].tolist()
nl_labels = dataset['distortion'].apply(lambda x: 0 if x == 'No Distortion' else 1).tolist()

en_data = pd.DataFrame({
    'text': en_texts,
    'label': en_labels,
})

nl_data = pd.DataFrame({
    'text': nl_texts,
    'label': nl_labels,
})

kt_data = pd.DataFrame({
    'text': kt_texts,
    'label': kt_labels,
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_datasets = []

en_splits = list(kf.split(en_data))
nl_splits = list(kf.split(nl_data))
kt_splits = list(kf.split(kt_data))

for fold in range(5):
    print(f"Fold {fold+1}")

    # Get train-test indices for English and Dutch
    en_train_idx, en_test_idx = en_splits[fold]
    _, nl_test_idx = nl_splits[fold]
    _, kt_test_idx = kt_splits[fold]

    en_train = Dataset.from_pandas(en_data.iloc[en_train_idx].reset_index(drop=True))
    en_test = Dataset.from_pandas(en_data.iloc[en_test_idx].reset_index(drop=True))
    nl_test = Dataset.from_pandas(nl_data.iloc[nl_test_idx].reset_index(drop=True))
    kt_test = Dataset.from_pandas(kt_data.iloc[kt_test_idx].reset_index(drop=True))

    dataset = DatasetDict({
        'train': en_train,
        'en': en_test,
        'nl': nl_test,
        'kt': kt_test,
    })

    fold_datasets.append(dataset)

dataset = fold_datasets[CURRENT_FOLD]

class_weights=(1/pd.Series(dataset['train']['label']).value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
print(class_weights)


tokenized_data = dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
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
        cls_state_dict = {'model.score': trainer.model.score.weight.detach().clone().cpu()}
        save_file(cls_state_dict, os.path.join(checkpoint_dir, 'score.safetensors'))
        return control  # Return control back to Trainer for normal operations


############################################################################################################################################################
## TRAINING THE MODEL

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    learning_rate = 1e-4,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 1,
    weight_decay = 0.01,
    report_to="none"
)
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data['train'],
    eval_dataset = tokenized_data['en'],
    processing_class = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
    callbacks=[CustomCheckpointCallback()]
)

trainer.train()

print("English test set")
trainer.evaluate(eval_dataset=tokenized_data['en'])

print("Translated English test set")
trainer.evaluate(eval_dataset=tokenized_data['nl'])

print("Kindertelefoon test set")
trainer.evaluate(eval_dataset=tokenized_data['kt'])
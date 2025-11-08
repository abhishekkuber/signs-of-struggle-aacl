'''
Script to run Instruction Tuning Llama preliminary for Table 1. 
Train : EN
Test : EN, NL, KT
'''

CURRENT_FOLD = 1

############################################################################################################################################################
## LIBRARIES

import numpy as np
import re
import pandas as pd
import torch

from sklearn.model_selection import KFold
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from sklearn.metrics import classification_report
from unsloth import FastLanguageModel, is_bfloat16_supported
from tqdm import tqdm

# FILL IN YOUR PATHS HERE
DATA_PATH = "..."
CACHE_DIR="..."
OUTPUT_DIR="..."


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################################################################################################################################
## FUNCTIONS

# formatting the dataset in the correct format before applying the chat template
def format_dataset(dataset, system_prompt):
    instruct_dataset = []
    for example in dataset:
        output_value = "Yes" if example["output"] == 1 else "No"
        instruct_dataset.append([
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example["input"]},
            {"from": "gpt", "value": output_value}
        ])
    return Dataset.from_dict({"conversations": instruct_dataset})

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


############################################################################################################################################################
## SETTING UP THE MODEL

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir=CACHE_DIR,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

############################################################################################################################################################
## DATA

# English data
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
    'input': en_texts[:10],
    'output': en_labels[:10],
})

nl_data = pd.DataFrame({
    'input': nl_texts[:10],
    'output': nl_labels[:10],
})

kt_data = pd.DataFrame({
    'input': kt_texts[:10],
    'output': kt_labels[:10],
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

system_prompt = """You are a psychologist trained to identify clear and explicit examples of cognitive distortions in English and Dutch text. Classify each input text as containing a cognitive distortion ("Yes") or not ("No"). Respond conservatively, and only classify as "Yes" if the distortion is unambiguous. Do not assume anything beyond the input text."""


dataset = DatasetDict({
    "train": format_dataset(dataset["train"], system_prompt),
    "en": format_dataset(dataset["en"], system_prompt),
    "nl": format_dataset(dataset["nl"], system_prompt),
    "kt": format_dataset(dataset["kt"], system_prompt),
})

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

dataset['train'] = standardize_sharegpt(dataset['train'])
dataset['en'] = standardize_sharegpt(dataset['en'])
dataset['nl'] = standardize_sharegpt(dataset['nl'])
dataset['kt'] = standardize_sharegpt(dataset['kt'])
dataset = dataset.map(formatting_prompts_func, batched = True,)



############################################################################################################################################################
## TRAINING

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir=OUTPUT_DIR,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(trainer, instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n", response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n")

trainer_stats = trainer.train()

############################################################################################################################################################
## TESTING

FastLanguageModel.for_inference(model)
pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"


# Kindertelefoon Test
print("Kindertelefoon Data")
test_data = dataset['kt']['conversations']
skips = 0
y_trues = []
y_preds = []
kt_preds = []

for example in tqdm(test_data):
  messages = example[:-1]
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True, temperature = 1.5, min_p = 0.1))
  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1)
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      kt_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = [1 if x == "Yes" else 0 for x in y_trues]
y_preds_binary = [1 if x == "Yes" else 0 for x in y_preds]

print(classification_report(y_trues_binary, y_preds_binary))


# English Test
print("English Data")
test_data = dataset['en']['conversations']
skips = 0
y_trues = []
y_preds = []
en_preds = []

for example in tqdm(test_data):
  messages = example[:-1]
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True, temperature = 1.5, min_p = 0.1))
  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1)
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      en_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = [1 if x == "Yes" else 0 for x in y_trues]
y_preds_binary = [1 if x == "Yes" else 0 for x in y_preds]

print(classification_report(y_trues_binary, y_preds_binary))


# Dutch Test
print("Dutch Data")
test_data = dataset['nl']['conversations']
skips = 0
y_trues = []
y_preds = []
nl_preds = []

for example in tqdm(test_data):
  messages = example[:-1]
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True, temperature = 1.5, min_p = 0.1))
  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1)
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      nl_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = [1 if x == "Yes" else 0 for x in y_trues]
y_preds_binary = [1 if x == "Yes" else 0 for x in y_preds]

print(classification_report(y_trues_binary, y_preds_binary))


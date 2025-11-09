'''
Script to run Llama Instruction Tuning (LLAMA IT) in Table 2. 
Train Data : EN + KT
Test Data : KT
'''

CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES

import numpy as np
import re
import pandas as pd
import torch
import unsloth

from sklearn.model_selection import KFold
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
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

english_data = pd.DataFrame({
    'input': en_texts,
    'output': en_labels,
})

dutch_data = pd.DataFrame({
    'input': kt_texts,
    'output': kt_labels,
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

    # Create DatasetDict for this fold
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    fold_datasets.append(dataset)

dataset = fold_datasets[CURRENT_FOLD]

system_prompt = """You are a psychologist trained to identify clear and explicit examples of cognitive distortions in English and Dutch text. Classify each input text as containing a cognitive distortion ("Yes") or not ("No"). Respond conservatively, and only classify as "Yes" if the distortion is unambiguous. Do not assume anything beyond the input text."""


dataset = DatasetDict({
    "train": format_dataset(dataset["train"], system_prompt),
    "test": format_dataset(dataset["test"], system_prompt),
})


tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

dataset['train'] = standardize_sharegpt(dataset['train'])
dataset['test'] = standardize_sharegpt(dataset['test'])
dataset = dataset.map(formatting_prompts_func, batched = True,)

############################################################################################################################################################
## TRAINING

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    args = SFTConfig(  ##
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim = "adamw_8bit",
        warmup_steps = 5,
        num_train_epochs = 1, 
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        packing=False,  ##
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        max_seq_length = max_seq_length,
        dataset_text_field="text",  
        report_to = "none",
    ),
)

trainer = train_on_responses_only(trainer, instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n", response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n")
trainer_stats = trainer.train()

############################################################################################################################################################
## TESTING

FastLanguageModel.for_inference(model)
pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"

test_data = dataset['test']['conversations']
skips = 0
y_trues = []
y_preds = []

for example in tqdm(test_data):
  messages = example[:-1]
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True, temperature = 1.5, min_p = 0.1))
  
  # We need to extract the response from the output using regex 
  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1)
      y_preds.append(y_pred)
      y_trues.append(example[-1]['content'])
  else:
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = [1 if x == "Yes" else 0 for x in y_trues]
y_preds_binary = [1 if x == "Yes" else 0 for x in y_preds]

print(classification_report(y_trues_binary, y_preds_binary))

print("Predictions: ", y_preds_binary)
print("Trues: ", y_trues_binary)
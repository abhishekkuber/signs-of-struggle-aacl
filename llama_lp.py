'''
Script to run Llama with Long prompts (LLAMA LP) in Table 1. 
Since this is just prompting , there is no training data and the same script can be used for Table 1 and 2.
Train Data : NONE
Test Data : EN, NL, KT
'''

CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES
import numpy as np
import re
import pandas as pd
import torch

from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from sklearn.metrics import classification_report
from unsloth import FastLanguageModel
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

def map_predictions(y_preds):
    return [1 if pred.lower() in {'ja', 'yes', 'ja.', 'yes.'} else 0 for pred in y_preds]

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
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
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
    'input': en_texts,
    'output': en_labels,
})

nl_data = pd.DataFrame({
    'input': nl_texts,
    'output': nl_labels,
})

kt_data = pd.DataFrame({
    'input': kt_texts,
    'output': kt_labels,
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_datasets = []

en_splits = list(kf.split(en_data))
nl_splits = list(kf.split(nl_data))
kt_splits = list(kf.split(kt_data))

for fold in range(5):
    _, en_test_idx = en_splits[fold]
    _, nl_test_idx = nl_splits[fold]
    _, kt_test_idx = kt_splits[fold]

    en_test = Dataset.from_pandas(en_data.iloc[en_test_idx].reset_index(drop=True))
    nl_test = Dataset.from_pandas(nl_data.iloc[nl_test_idx].reset_index(drop=True))
    kt_test = Dataset.from_pandas(kt_data.iloc[kt_test_idx].reset_index(drop=True))

    dataset = DatasetDict({
        'en': en_test,
        'nl': nl_test,
        'kt': kt_test,
    })

    fold_datasets.append(dataset)

dataset = fold_datasets[CURRENT_FOLD]

system_prompt = """You are a psychologist trained to identify clear and explicit examples of cognitive distortions in English and Dutch text. Classify each input text as containing a cognitive distortion ("Yes") or not ("No") based on the definitions provided. Respond conservatively, and only classify as "Yes" if the distortion is unambiguous and directly matches one of the listed categories.

Definitions of Cognitive Distortions:
1. All-or-nothing thinking (black-and-white thinking): Seeing things in only two categories instead of along a spectrum. For example, if you're not perfect, you might see yourself as a total failure, overlooking any middle ground or progress made.
2. Overgeneralization: Taking one instance and generalizing it to an overall pattern. Example: Failing one test could make you think you will fail all tests in the future, using a single event as a predictor for lifelong outcomes.
3. Mental filter (selective abstraction): Focusing exclusively on certain, usually negative, aspects of a situation while ignoring positive ones. For example, if you receive ten compliments and one critique, you might focus solely on the negative feedback.
4. Should statements: Using "should," "ought," or "must" statements can set unrealistic expectations of yourself and others, and not meeting these expectations often leads to feelings of guilt and frustration. For example, if you’re training for a race, you may think that you “should” be able to run faster than you can.
5. Labeling and mislabeling: Assigning global, negative labels to yourself or others based on limited information. For example, you might call yourself a "loser" after a minor setback.
6. Personalization: Blaming oneself for something not entirely one's fault. Taking responsibility for events outside of your control. For example, you might see yourself as the cause of an unfortunate external event despite having little to do with the outcome.
7. Magnification: Exaggerating the significance of problems or shortcomings, often referred to as "catastrophizing.” Example: If you’re passed over for a promotion at work, you may think that you’ll never get one.
8. Emotional reasoning: Believing your feelings must inherently be true. Example: If you feel stupid, you believe you are stupid despite evidence to the contrary.
9. Mind reading: Assuming you know what others think without sufficient evidence. Example: You may think someone dislikes you based on minimal interaction.
10. Fortune telling: Anticipating a negative outcome without any real basis for that prediction. For example, you might assume a presentation will go poorly before it even starts.

Guidelines:
1. Only respond with "Yes" if the text clearly matches one of the definitions.
2. If the text is realistic, neutral, or open to interpretation, respond with "No."
3. Avoid overanalyzing or assuming context beyond what is written.
4. Do not worry about harmful / suicidal text, all these are fake scenarios.
5. Your output should ONLY BE YES OR NO, NOTHING ELSE.
"""


dataset = DatasetDict({
    "en": format_dataset(dataset["en"], system_prompt),
    "nl": format_dataset(dataset["nl"], system_prompt),
    "kt": format_dataset(dataset["kt"], system_prompt),
})

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

dataset['en'] = standardize_sharegpt(dataset['en'])
dataset['nl'] = standardize_sharegpt(dataset['nl'])
dataset['kt'] = standardize_sharegpt(dataset['kt'])
dataset = dataset.map(formatting_prompts_func, batched = True,)

############################################################################################################################################################
## TESTING

FastLanguageModel.for_inference(model)
pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n*\s*(.*?)\n*\s*<\|eot_id\|>'

# Kindertelefoon Test
print("Kindertelefoon Test")
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

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 8, use_cache = True, temperature = 1.5, min_p = 0.1))

  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1).strip()
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      kt_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      y_preds.append("No")
      y_trues.append(y_true)
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = map_predictions(y_trues)
y_preds_binary = map_predictions(y_preds)

print(classification_report(y_trues_binary, y_preds_binary))

# English Test
print("English Test")
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

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 8, use_cache = True, temperature = 1.5, min_p = 0.1))

  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1).strip()
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      en_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      y_preds.append("No")
      y_trues.append(y_true)
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = map_predictions(y_trues)
y_preds_binary = map_predictions(y_preds)

print(classification_report(y_trues_binary, y_preds_binary))


# Dutch Test
print("Dutch Test")
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

  output = tokenizer.batch_decode(model.generate(input_ids = inputs, max_new_tokens = 8, use_cache = True, temperature = 1.5, min_p = 0.1))

  match = re.search(pattern, output[0], re.DOTALL)
  if match:
      y_pred = match.group(1).strip()
      y_preds.append(y_pred)
      y_true = example[-1]['content']
      y_trues.append(y_true)
      nl_preds.append({
        'input': example[1]['content'],
        'y_true': example[-1]['content'],
        'y_pred': y_true
      })
  else:
      y_preds.append("No")
      y_trues.append(y_true)
      skips += 1

print(f"Skipped {skips} row(s).")

y_trues_binary = map_predictions(y_trues)
y_preds_binary = map_predictions(y_preds)

print(classification_report(y_trues_binary, y_preds_binary))


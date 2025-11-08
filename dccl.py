'''
Script to run DCCL in Table 2. 
Train : EN + KT
Test : KT
'''

CURRENT_FOLD = 0

############################################################################################################################################################
## LIBRARIES
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import json
import evaluate
import os

from transformers import AutoTokenizer, TrainingArguments, Trainer, XLMRobertaForSequenceClassification, AutoConfig
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tqdm import tqdm

# FILL IN YOUR PATHS HERE
DATA_PATH = "..."
CACHE_DIR="..."
OUTPUT_DIR="..."

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################# FIRST TRAINING LOOP ##########################################################################################

############################################################################################################################################################
## FUNCTIONS
def encode_batch(batch):
  return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    print(classification_report(labels, predictions))
    return {**accuracy, **f1}

# COPIED FROM https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
# Done so as to avoid making a new container with InfoNCE installed
class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

############################################################################################################################################################
## DATA

# loading the english data
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
    'domain': [0] * len(english_texts)  # Label 0 for English data

})

dutch_data = pd.DataFrame({
    'text': dutch_texts,
    'label': dutch_labels,
    'domain': [1] * len(dutch_texts)  # Label 1 for Dutch data
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
    
    train_dataset = Dataset.from_pandas(train_data_combined)
    test_dataset = Dataset.from_pandas(dutch_test)

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    fold_datasets.append(dataset)

dataset = fold_datasets[CURRENT_FOLD]

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir = CACHE_DIR)

dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "domain"])

############################################################################################################################################################
## MODEL
class PerturbationLayer(nn.Module):
    def __init__(self, hidden_dim=768, sigma=0.02):
        super().__init__()
        self.perturbation = nn.Parameter(torch.zeros(hidden_dim))
        self.sigma = sigma

    def forward(self, x):
        delta = self.sigma * self.perturbation
        return x + delta

class DCCL(XLMRobertaForSequenceClassification):
    def __init__(self, config, alpha_domain_classification=1e-3, lambda_contrastive=3e-2, beta_consistency=5, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels

        self.perturbation_layer = PerturbationLayer()

        self.domain_classifier = nn.Linear(768, 2)
        self.domain_classification_loss = nn.CrossEntropyLoss()
        self.alpha_domain_classification = alpha_domain_classification

        self.projection = nn.Linear(768, 256)
        self.contrastive_loss = InfoNCE()
        self.lambda_contrastive = lambda_contrastive

        self.classifier = nn.Linear(768, self.num_labels)
        self.classification_loss = nn.CrossEntropyLoss()

        self.consistency_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.beta_consistency = beta_consistency


    def forward(self, input_ids=None, attention_mask=None, labels=None, domain=None):
        # final loss = classification + alpha*domain_classification + lambda*contrastive * beta*consistency
        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        orig_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply learnable perturbation
        perturbed_embedding = self.perturbation_layer(orig_embedding)

        # Domain Classification Loss
        domain_logits = self.domain_classifier(perturbed_embedding)
        domain_cls_loss = self.domain_classification_loss(domain_logits.view(-1, 2), domain.view(-1))


        # Distortion classification
        logits_orig = self.classifier(orig_embedding)
        logits_pert = self.classifier(perturbed_embedding)

        cls_loss = self.classification_loss(logits_orig.view(-1, 2), labels.view(-1))

        # Consistency loss
        consist_loss = self.consistency_loss(F.log_softmax(logits_pert, dim=-1), F.log_softmax(logits_orig, dim=-1))

        # Contrastive loss
        orig_embedding_proj = self.projection(orig_embedding)
        perturbed_embedding_proj = self.projection(perturbed_embedding)

        contrast_loss = self.contrastive_loss(orig_embedding_proj, perturbed_embedding_proj)

        total_loss = cls_loss + (self.alpha_domain_classification * (-domain_cls_loss)) + self.lambda_contrastive * contrast_loss + self.beta_consistency * consist_loss

        return {
            "loss": total_loss,
            "classification loss": cls_loss,
            "domain classification loss": domain_cls_loss,
            "contrastive loss": contrast_loss,
            "consistency loss": consist_loss,
            "logits": logits_orig
            }

############################################################################################################################################################
## TRAINING
model = DCCL.from_pretrained("xlm-roberta-base", cache_dir = CACHE_DIR)

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/dccl_trainer_fold_{CURRENT_FOLD}",
    eval_strategy="epoch",
    save_strategy="no",
    save_total_limit=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay = 0.01,
    learning_rate=1e-5,
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer
)


print("\nBefore Training!")
trainer.evaluate()

print("\nSTARTING FIRST TRAINING LOOP")
trainer.train()

print("\nAfter Training!")
trainer.evaluate()

model.save_pretrained(f"{OUTPUT_DIR}/dccl_encoder_fold_{CURRENT_FOLD}")

############################################# SECOND TRAINING LOOP #########################################################################################
# Now in the second training loop, we only train the distortion classifier and the components associated with it, the rest of the model is frozen. 
############################################################################################################################################################
## MODEL

# We change the model's forward pass
class PerturbationLayer(nn.Module):
    def __init__(self, hidden_dim=768, sigma=0.02):
        super().__init__()
        self.perturbation = nn.Parameter(torch.zeros(hidden_dim))
        self.sigma = sigma

    def forward(self, x):
        delta = self.sigma * self.perturbation
        return x + delta

class DCCL(XLMRobertaForSequenceClassification):
    def __init__(self, config, alpha_domain_classification=1e-3, lambda_contrastive=3e-2, beta_consistency=5, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels

        # Learnable perturbations
        self.perturbation_layer = PerturbationLayer()

        # Classifier for predicting which language the embedding (normal or perturbed) belongs to
        self.domain_classifier = nn.Linear(768, 2)
        self.domain_classification_loss = nn.CrossEntropyLoss()
        self.alpha_domain_classification = alpha_domain_classification

        # Projection head, used for contrastive loss (MLP in the diagram)
        self.projection = nn.Linear(768, 256)
        self.contrastive_loss = InfoNCE()
        self.lambda_contrastive = lambda_contrastive

        # Classifier for predicting whether a piece of text contains a distortion or not
        self.classifier = nn.Linear(768, self.num_labels)
        self.classification_loss = nn.CrossEntropyLoss()

        self.consistency_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.beta_consistency = beta_consistency


    def forward(self, input_ids=None, attention_mask=None, labels=None, domain=None):
        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        orig_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(orig_embedding)

        loss = self.classification_loss(logits.view(-1, 2), labels.view(-1))

        return {"loss": loss, "logits": logits}

torch.cuda.empty_cache()
del model

model_name = f"{OUTPUT_DIR}/dccl_encoder_fold_{CURRENT_FOLD}"
model = DCCL.from_pretrained(model_name, cache_dir=CACHE_DIR)

############################################################################################################################################################
## DATA
dataset = fold_datasets[CURRENT_FOLD]
dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

############################################################################################################################################################
## TRAINING

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/classification_trainer_fold_{CURRENT_FOLD}",
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    learning_rate=2e-5,                 # Learning rate
    per_device_train_batch_size=8,      # Batch size for training
    per_device_eval_batch_size=8,       # Batch size for evaluation
    num_train_epochs=2,                 # Number of training epochs
    weight_decay=0.01,                  # Weight decay for regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

print("\nBefore Training!")
trainer.evaluate()

print("\nSTARTING SECOND TRAINING LOOP")
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

model.save_pretrained(f"{OUTPUT_DIR}/dccl_classifier_fold_{CURRENT_FOLD}")
# Code for doing the McNemar's test on the predictions of different models

import numpy as np
from mlxtend.evaluate import mcnemar_table, mcnemar
from statsmodels.stats.multitest import multipletests
from itertools import combinations


# Fill them with the predictions of the models for 5 folds, as a list of zero and ones. 
# Replace them with the actual predictions from your models.
adapters_pred = np.array([0, 1, 0, 1, 0])
dccl_pred = np.array([0, 1, 0, 1, 0])
xlmr_empath_pred = np.array([0, 1, 0, 1, 0])
xlmr_ft_pred = np.array([0, 1, 0, 1, 0])
llama_ft_pred = np.array([0, 1, 0, 1, 0])
llama_it_pred = np.array([0, 1, 0, 1, 0])

y_true = np.array([0, 1, 0, 1, 0])  # Replace with your true labels

def get_predictions(model_name):
    if model_name == "Adapters":
        return adapters_pred
    elif model_name == "DCCL Correct":
        return dccl_pred
    elif model_name == "XLMR Empath":
        return xlmr_empath_pred
    elif model_name == "XLMR Finetuning":
        return xlmr_ft_pred
    elif model_name == "Llama Classification":
        return llama_ft_pred
    elif model_name == "Llama Instruction Tuning":
        return llama_it_pred

def get_mcnemar(model_1, model_2, y_true, y_pred_1, y_pred_2):
    tb = np.array(mcnemar_table(y_target=y_true, y_model1=y_pred_1, y_model2=y_pred_2))
    chi2, p = mcnemar(ary=tb, corrected=True)
    return p

raw_p_values = []
# Put in the names of the models you want to compare
comb = combinations(["Adapters", "DCCL Correct", "XLMR Empath"], 2)
all_combs = list(comb)
for i in all_combs:
    model_1, model_2 = i
    y_pred_1 = get_predictions(model_1)
    y_pred_2 = get_predictions(model_2)
    p = get_mcnemar(model_1, model_2, y_true, y_pred_1, y_pred_2)
    print(f"{model_1} vs {model_2}: p-value = {p}, reject = {p < 0.05}")
    raw_p_values.append(p)

# Apply the Bonferroni correction 
bonferroni_corrected = multipletests(raw_p_values, alpha=0.05, method='bonferroni')

# # Show results
a = []
print("Bonferroni corrected p-values:")
for i, (pair, p_val, corrected_p, reject) in enumerate(zip(all_combs, raw_p_values, bonferroni_corrected[1], bonferroni_corrected[0])):
    # Or you can print the corrected p-values and compare them to the original threshold level.
    print(f"{pair[0]} vs {pair[1]} & {corrected_p:.4f} & {corrected_p < 0.05} \\\\")
    
    # # You can either print the original p-values and compare them to the threshold which becomes p / number of comparisons. In this case, it is 0.05 / 3 = 0.0167.
    # print(f"{pair[0]} vs {pair[1]} & {p_val:.4f} & {p_val < 0.0167} \\\\")
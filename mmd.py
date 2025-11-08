# Once you have trained the model, please run this script to calculate the MMD scores. You will need the trained model, so just append 
# this piece of code to the end of your training script. 


# Function to get the CLS embeddings for a list of texts for a model. 
def get_cls_embeddings(texts, model, tokenizer):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts):
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get CLS embedding (output of the last layer)
            cls_embedding = model(**inputs)  # This only returns the CLS embedding

            # # if that doesnt work, use the following:
            # outputs = model.roberta(**inputs, output_hidden_states=True, return_dict=True)
            # cls_embedding = outputs.hidden_states[-1][:, 0, :]  # CLS token from last layer

            # Append the CLS embedding
            embeddings.append(cls_embedding.cpu().numpy())

    return np.vstack(embeddings)


# Get the embeddings for the English and Dutch datasets
english_embeddings = get_cls_embeddings(english_data['text'].tolist(), model, tokenizer)
dutch_embeddings = get_cls_embeddings(dutch_data['text'].tolist(), model, tokenizer)


en_distorted = []
en_not_distorted = []
kt_distorted = []
kt_not_distorted = []

# Separate the embeddings based on the distortion and dataset labels
for i, j in zip(english_embeddings, english_data['label'].values):
  if j == 1:
    en_distorted.append(i)
  elif j == 0:
    en_not_distorted.append(i)
  else:
    print("INVALID")

unique, counts = np.unique(english_data['label'].values, return_counts=True)
print(dict(zip(unique, counts)))
print(len(en_distorted))
print(len(en_not_distorted))

for i, j in zip(dutch_embeddings, dutch_data['label'].values):
  if j == 1:
    kt_distorted.append(i)
  elif j == 0:
    kt_not_distorted.append(i)
  else:
    print("INVALID")

unique, counts = np.unique(dutch_data['label'].values, return_counts=True)
print(dict(zip(unique, counts)))
print(len(kt_distorted))
print(len(kt_not_distorted))

en_distorted = np.stack(en_distorted)
kt_distorted = np.stack(kt_distorted)
en_not_distorted = np.stack(en_not_distorted)
kt_not_distorted = np.stack(kt_not_distorted)




# The function to calculate MMD has been copied from kaggle
# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

# Since MMD requires the same number of samples from both distributions, we will subsample the data.
def calculate_subsampled_mmd(x, y, kernel, n_repeats=10, seed=42):
    np.random.seed(seed)
    min_n = min(x.shape[0], y.shape[0])
    mmd_scores = []

    for _ in trange(n_repeats, desc="Subsampled MMD"):
        idx_x = np.random.choice(x.shape[0], min_n, replace=False)
        idx_y = np.random.choice(y.shape[0], min_n, replace=False)
        x_sub = x[idx_x].to(device)
        y_sub = y[idx_y].to(device)
        mmd = MMD(x_sub, y_sub, kernel)
        mmd_scores.append(mmd.item())

    return np.mean(mmd_scores), np.std(mmd_scores)


# Calculate MMD scores between EN distorted and KT distorted
mmd_distorted_mean_multiscale, mmd_distorted_std_multiscale = calculate_subsampled_mmd(torch.from_numpy(en_distorted), torch.from_numpy(kt_distorted), kernel="multiscale")
print(f"Distorted (EN vs KT): {mmd_distorted_mean_multiscale:.2f} ± {mmd_distorted_std_multiscale:.2f}")

# Calculate MMD scores between EN not distorted and KT not distorted
mmd_not_distorted_mean_multiscale, mmd_not_distorted_std_multiscale = calculate_subsampled_mmd(torch.from_numpy(en_not_distorted), torch.from_numpy(kt_not_distorted), kernel="multiscale")
print(f"Not Distorted (EN vs KT): {mmd_not_distorted_mean_multiscale:.2f} ± {mmd_not_distorted_std_multiscale:.2f}")

# Calculate MMD scores between EN distorted and EN not distorted
mmd_en_mean_multiscale, mmd_en_std_multiscale = calculate_subsampled_mmd(torch.from_numpy(en_distorted), torch.from_numpy(en_not_distorted), kernel="multiscale")
print(f"EN (Distorted vs Not Distorted): {mmd_en_mean_multiscale:.2f} ± {mmd_en_std_multiscale:.2f}")

# Calculate MMD scores between KT distorted and KT not distorted
mmd_kt_mean_multiscale, mmd_kt_std_multiscale = calculate_subsampled_mmd(torch.from_numpy(kt_distorted), torch.from_numpy(kt_not_distorted), kernel="multiscale")
print(f"KT (Distorted vs Not Distorted): {mmd_kt_mean_multiscale:.2f} ± {mmd_kt_std_multiscale:.2f}")
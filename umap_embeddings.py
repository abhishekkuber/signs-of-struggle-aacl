# Code to visualize UMAP embeddings for English and Dutch datasets. 
# Since this needs the trained models and the data, add this code to the end of the training scripts. 

import matplotlib.pyplot as plt
import numpy as np
import umap


# Combine embeddings
all_embeddings = np.vstack((english_embeddings, dutch_embeddings))
all_labels = np.concatenate((english_data['label'].values, dutch_data['label'].values))
all_domains = np.concatenate([np.array([0] * len(english_data)), np.array([1] * len(dutch_data))])


###### PLOTTING BOTH DOMAIN EMBEDDINGS ######
# Apply UMAP for dimensionality reduction
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

# Assign colors based on language and label
colors = {
    (0, 1): "red",    # English Label 1
    (0, 0): "blue",   # English Label 0
    (1, 1): "green",  # Dutch Label 1
    (1, 0): "orange"  # Dutch Label 0
}
point_colors = [colors[(domain, label)] for domain, label in zip(all_domains, all_labels)]

# Plot the UMAP visualization
plt.figure(figsize=(8,6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=point_colors, alpha=0.7)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


###### PLOTTING EN EMBEDDINGS ######
# Apply UMAP on only the English embeddings
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(english_embeddings)

# Assign colors based on label
colors = {
    1: "red",   
    0: "blue"   
}
point_colors = [colors[label] for label in english_data['label'].values]

# Plot the UMAP visualization
plt.figure(figsize=(8,6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=point_colors, alpha=0.7)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


###### PLOTTING KT EMBEDDINGS ######
# Apply UMAP on only the English embeddings
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(dutch_embeddings)

colors = {
    1: "green",   
    0: "orange"   
}
point_colors = [colors[label] for label in dutch_data['label'].values]

# Plot the UMAP visualization
plt.figure(figsize=(8,6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=point_colors, alpha=0.7)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
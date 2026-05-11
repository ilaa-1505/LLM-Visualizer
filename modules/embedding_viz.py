import torch
import numpy as np

WORD_CATEGORIES = {
    "royalty": ["king", "queen", "prince", "princess", "throne", "crown"],
    "animals": ["dog", "cat", "lion", "eagle", "shark", "wolf"],
    "places": ["paris", "london", "delhi", "tokyo", "rome", "berlin"],
    "food": ["pizza", "rice", "bread", "curry", "pasta", "sushi"],
    "tech": ["code", "model", "data", "neural", "algorithm", "computer"],
    "emotions": ["happy", "sad", "angry", "fear", "love", "hate"],
    "nature": ["river", "mountain", "forest", "ocean", "desert", "sky"]
}

def get_word_embeddings(model, tokenizer):
    embedding_matrix = model.transformer.wte.weight
    
    words = []
    categories = []
    vectors = []

    for category, word_list in WORD_CATEGORIES.items():
        for word in word_list:
            token_id = tokenizer.encode(word)[0]
            vector = embedding_matrix[token_id].detach().numpy()
            words.append(word)
            categories.append(category)
            vectors.append(vector)

    return words, categories, np.array(vectors)


from sklearn.decomposition import PCA
import umap

def reduce_dimensions(vectors, method="PCA", n_components=3):
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    else:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    
    return reducer.fit_transform(vectors)
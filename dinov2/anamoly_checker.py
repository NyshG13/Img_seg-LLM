import numpy as np
from scipy.spatial.distance import cosine

# Load normal embeddings once
normal_embeddings = np.load("normal_data/normal_embeddings.npy")

if normal_embeddings.shape == (1,) and isinstance(normal_embeddings[0], (list, np.ndarray)):
    normal_embeddings = np.array(normal_embeddings[0])

def is_anomalous(new_embedding, threshold=0.85):
    """
    Compares a new embedding to saved normal ones.
    Returns True if it is sufficiently different (i.e., anomaly).
    """
    new_embedding = np.squeeze(new_embedding)
    if new_embedding.ndim != 1:
        raise ValueError(f"new_embedding must be 1-D but got shape {new_embedding.shape}")

    for normal_emb in normal_embeddings:
        normal_emb = np.squeeze(normal_emb)
        if normal_emb.ndim != 1:
            raise ValueError(f"normal_emb must be 1-D but got shape {normal_emb.shape}")

        sim = 1 - cosine(new_embedding, normal_emb)

        if sim > threshold:
            return False   # Not anomalous, it's similar to something normal
    return True
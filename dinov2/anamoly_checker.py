import numpy as np
from scipy.spatial.distance import cosine

# Load normal embeddings once
normal_embeddings = np.load("normal_data/normal_embeddings.npy")

def is_anomalous(new_embedding, threshold=0.85):
    """
    Compares a new embedding to saved normal ones.
    Returns True if it is sufficiently different (i.e., anomaly).
    """
    for normal_emb in normal_embeddings:
        similarity = 1 - cosine(new_embedding, normal_emb)
        if similarity > threshold:
            return False  # Not anomalous, it's similar to something normal
    return True
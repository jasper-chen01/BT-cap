"""
Helper utility functions
"""
import numpy as np
from typing import List, Dict


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of embeddings"""
    emb1_norm = normalize_embeddings(embeddings1)
    emb2_norm = normalize_embeddings(embeddings2)
    return np.dot(emb1_norm, emb2_norm.T)


def majority_vote(annotations: List[str], weights: List[float] = None) -> str:
    """Get majority vote from list of annotations, optionally weighted"""
    if not annotations:
        return "Unknown"
    
    if weights is None:
        weights = [1.0] * len(annotations)
    
    # Count weighted votes
    votes = {}
    for ann, weight in zip(annotations, weights):
        votes[ann] = votes.get(ann, 0) + weight
    
    # Return annotation with highest vote
    return max(votes.items(), key=lambda x: x[1])[0]


def calculate_confidence(annotations: List[str], predicted: str) -> float:
    """Calculate confidence score as fraction of annotations matching prediction"""
    if not annotations:
        return 0.0
    return annotations.count(predicted) / len(annotations)

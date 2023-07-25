import torch
import torch.nn as nn


class EmbedUtil:
    def __init__(self, embedding_tensor):
        self.embedding_tensor = embedding_tensor
        self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0], embedding_dim=embedding_tensor.shape[1])
        self.embedding.weight = torch.nn.Parameter(embedding_tensor)

    def embed(self, input_tensor):
        embedded = self.embedding(input_tensor)  # (batch, sequence, dim)
        return embedded

    # Find the index of the closet vector to the target.
    # uni_embedding_vectors should be in (char size, embedding dim)
    # target_vector should be in (embedding dim)
    def find_closet_vector_idx(self, target_vector):
        distances = torch.cdist(target_vector, self.embedding_tensor, p=2).squeeze(0)
        nearest_index = torch.argmin(distances, dim=2)
        return nearest_index


def mse_distance(x, y):
    return ((x - y) ** 2).mean().sqrt()


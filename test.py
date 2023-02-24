import torch
import torch.nn as nn

import pytest
from optim_utils import nn_project

@pytest.fixture
def sample_input():
    return torch.tensor([[0.1, 0.2], [-0.3, -0.4]])

def nn_project(nn_indices: Tensor, embedding_layer: nn.Linear) -> Tensor:
    # Cast nn_indices to the same dtype as embedding_layer.weight
    nn_indices = nn_indices.to(embedding_layer.weight.dtype)

    # Project the indices onto the embedding space
    projected_embeds = embedding_layer(nn_indices)

    # Normalize the projected embeddings
    norm = projected_embeds.norm(p=2, dim=1, keepdim=True)
    normalized_embeds = projected_embeds.div(norm)

    return normalized_embeds

import torch
import pytest
from optim_utils import nn_project

@pytest.fixture
def sample_input():
    return torch.tensor([[0.1, 0.2], [-0.3, -0.4]])

def test_nn_project(sample_input):
    embedding_layer = nn.Linear(2, 2)
    expected_output = torch.tensor([[0.4, 0.8], [0.0, 0.0]])
    output = nn_project(sample_input, embedding_layer)
    assert torch.allclose(output, expected_output, atol=1e-2)

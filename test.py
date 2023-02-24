import torch
import pytest
from my_module import nn_project

@pytest.fixture
def sample_input():
    return torch.tensor([[0.1, 0.2], [-0.3, -0.4]])

def nn_project(sample_input):
    expected_output = torch.tensor([[0.4, 0.8], [0.0, 0.0]])
    output = nn_project(sample_input)
    assert torch.allclose(output, expected_output)

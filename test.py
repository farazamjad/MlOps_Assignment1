import torch
from transformers import AutoTokenizer

def test_decode_ids():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    input_ids = torch.tensor([[101, 2023, 2003, 1037, 999, 102], [101, 2023, 2003, 1037, 2000, 102]])
    decoded = decode_ids(input_ids, tokenizer)
    assert decoded == ['[CLS] the quick brown [UNK] [SEP]', '[CLS] the quick brown fox [SEP]']

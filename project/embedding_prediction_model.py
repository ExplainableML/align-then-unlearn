import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class EmbeddingPredictionTransformer(nn.Module):
    @classmethod
    def from_config(cls, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, num_layers: int, dropout: float, **kwargs):
        return cls(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = nn.Linear(self.input_dim, hidden_dim)
        # self.positional_encoding = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output = nn.Linear(hidden_dim, self.output_dim)

    def forward(
        self, x: Tensor
    ) -> Tensor:
        x = self.input(x)  # [batch_size, seq_length, hidden_dim]
        # x = self.positional_encoding(x)  # Apply positional encoding
        # x = torch.nn.functional.normalize(x, dim=-1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(
            x.device
        )
        x = self.transformer(
            tgt=x, memory=x, tgt_mask=tgt_mask
        )  # [batch_size, seq_length, hidden_dim]
        x = self.output(x)  # [batch_size, seq_length, output_dim]
        # x = torch.nn.functional.normalize(x, dim=-1)
        return x

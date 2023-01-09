"""This module contains the Decoder."""

import torch as T
import torch.nn as nn

from transfomer import attn

class Decoder(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, n_decoder_blocks: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.decoder_blocks = []

  def forward(self) -> T.Tensor:
    ...

class DecoderBlock(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.multihead_self_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.multihead_cross_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.feedfwd = nn.Linear(d_model, d_model)

  def forward(self, encoding: T.Tensor, outputs: T.Tensor) -> T.Tensor:
    ...

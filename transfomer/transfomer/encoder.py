"""This module contains the encoder."""

from typing import Sequence
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from transfomer import attn

class Encoder(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, n_encoder_blocks: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.n_encoder_blocks = n_encoder_blocks
    self.lut = None
    self.encoder_blocks = [EncoderBlock(d_model, d_k, d_v, n_heads) for _ in range(n_encoder_blocks)]

  def forward(self, tokens: Sequence) -> T.Tensor:
    embedding = T.randn(8, len(tokens), self.d_model)
    for block in self.encoder_blocks:
      embedding = block(embedding)
    return embedding

class EncoderBlock(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.multihead_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.feedfwd = nn.Linear(d_model, d_model)

  def forward(self, E: T.Tensor) -> T.Tensor:
    """
    Encoder block takes in an encoding of dims:
     (batch_size, token_len, embedding_dim)
    """
    self_attn = self.multihead_attn(Q=E, K=E, V=E)
    sublayer = F.layer_norm(self_attn + E, normalized_shape=(self.d_model, ))
    return F.layer_norm(self.feedfwd(sublayer) + sublayer, normalized_shape=(self.d_model, ))

if __name__ == "__main__":
  batch_size, n_tokens = 8, 69
  d_model, d_k, d_v, d_o, n_heads, n_encoders = 512, 64, 64, 64, 8, 6
  encoder = Encoder(d_model, d_k, d_v, n_heads, n_encoders)
  output = encoder([1,2,3])

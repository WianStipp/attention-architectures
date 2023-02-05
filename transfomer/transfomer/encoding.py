"""This module contains the encoder."""

from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from transfomer import attn, positional_encoding

MAX_SEQUENCE_LENGTH = 10000

class Feedforward(nn.Module):
  """
  Feedforward network as described in Vaswani et al.
  FFN(x) = max(0, xW_1 + b)W_2 + b
  """
  def __init__(self, d_model: int) -> None:
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_model)
    self.linear2 = nn.Linear(d_model, d_model)

  def forward(self, inputs: T.Tensor) -> T.Tensor:
    return self.linear2(F.relu(self.linear1(inputs)))

class Encoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int, n_heads: int, n_encoder_blocks: int) -> None:
    super().__init__()
    self.lut = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    self.encoder_blocks = [EncoderBlock(d_model, d_k, d_v, n_heads) for _ in range(n_encoder_blocks)]
    self.positional_encoder = positional_encoding.PositionalEncoding(d_model)

  def forward(self, tokens: T.Tensor, attention_mask: Optional[T.Tensor] = None) -> T.Tensor:
    """Do a forward pass through the encoder
    Args:
      tokens: A sequence of tokens, with shape: (batch_size, n_tokens)
    """
    embedding = self.lut(tokens) # batch_size * n_tokens * embedding dim
    embedding += self.positional_encoder(embedding)
    for block in self.encoder_blocks:
      embedding = block(embedding, attention_mask)
    return embedding

class EncoderBlock(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.multihead_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.feedfwd = Feedforward(d_model)

  def forward(self, E: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
    """
    Encoder block takes in an encoding of dims:
     (batch_size, token_len, embedding_dim)
    """
    self_attn = self.multihead_attn(Q=E, K=E, V=E, mask=mask)
    sublayer = F.layer_norm(self_attn + E, normalized_shape=(self.d_model, ))
    return F.layer_norm(self.feedfwd(sublayer) + sublayer, normalized_shape=(self.d_model, ))


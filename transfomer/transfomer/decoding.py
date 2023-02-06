"""This module contains the Decoder."""

from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from transfomer import attn, positional_encoding, encoding

class Decoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, dim_feedfwd: int, d_k: int, d_v: int, n_heads: int, n_decoder_blocks: int) -> None:
    super().__init__()
    self.lut = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, d_k, d_v, n_heads, dim_feedfwd) for _ in range(n_decoder_blocks)])
    self.positional_encoder = positional_encoding.PositionalEncoding(d_model)

  def forward(self, encoding: T.Tensor, tokens: T.Tensor, input_mask: Optional[T.Tensor] = None, target_mask: Optional[T.Tensor] = None) -> T.Tensor:
    """Do a single forward pass through the decoder."""
    embedding = self.lut(tokens)
    embedding += self.positional_encoder(embedding)
    for block in self.decoder_blocks:
      embedding = block(encoding, embedding, input_mask, target_mask)
    return embedding

class DecoderBlock(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, dim_feedfwd: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.multihead_self_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.multihead_cross_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.feedfwd = encoding.Feedforward(d_model, dim_feedfwd)
    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.1)
    self.dropout3 = nn.Dropout(0.1)

  def forward(self, encoding: T.Tensor, prev_outputs: T.Tensor, input_mask: Optional[T.Tensor] = None, target_mask: Optional[T.Tensor] = None) -> T.Tensor:
    self_attn = self.multihead_self_attn(Q=prev_outputs, K=prev_outputs, V=prev_outputs, mask=target_mask)
    sublayer = F.layer_norm(self.dropout1(self_attn) + prev_outputs, normalized_shape=(self.d_model, ))
    cross_attn = self.multihead_cross_attn(Q=sublayer, K=encoding, V=encoding, mask=None)
    sublayer2 = F.layer_norm(self.dropout2(cross_attn) + sublayer, normalized_shape=(self.d_model, ))
    dense_out = self.feedfwd(sublayer2)
    return F.layer_norm(self.dropout3(dense_out)+ sublayer2, normalized_shape=(self.d_model, ))

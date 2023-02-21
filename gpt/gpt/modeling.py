"""Thos module contains the model code for GPT."""

from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from gpt import gpt_config


class GPT(nn.Module):
  """Simple GPT-like decoder transformer model"""
  def __init__(self, config: gpt_config.GPTConfig) -> None:
    super().__init__()
    self.config = config
    self.lut = nn.Embedding(self.config.vocab_size, self.config.d_model)
    self.decoder_blocks = nn.ModuleList([GPTDecoderBlock.from_gptconfig(self.config) for _ in range(config.n_decoder_blocks)])
    # TODO: add positional encoding
    self.positional_encoder = lambda x: x

  def forward(self, tokens: T.Tensor, padding_mask: Optional[T.Tensor] = None):
    """Do a single pass through GPT.

    Args:
      tokens: (batch_size, sequence_length) shape of tokens.
      padding_mask: Padding mask (causal mask will be applied internally)"""
    embeddings = self.lut(tokens)
    embeddings += self.positional_encoder(embeddings)
    for block in self.decoder_blocks:
      embeddings = block(embeddings, padding_mask)
    return None, (embeddings**2).mean()
    #TODO: next token prediction head


class GPTDecoderBlock(nn.Module):
  """Decoder block that does not compute cross attention with any encodings."""
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, dim_feedfwd: int, dropout_rate: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_rate, batch_first=True)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)

  def forward(self, embeddings: T.Tensor, padding_mask: T.Tensor) -> T.Tensor:
    """The decoder block takes in token embeddings and enriches them with a series
    of transfomations including causal self-attention."""
    causal_mask = make_causal_mask(padding_mask.shape[-1]).to('cuda')
    self_attn_out, _ = self.self_attn(embeddings, embeddings, embeddings, key_padding_mask=padding_mask, attn_mask=causal_mask)
    layer1 = F.layer_norm(self_attn_out + embeddings, normalized_shape=(self.d_model, ))
    return layer1

  @classmethod
  def from_gptconfig(cls, config: gpt_config.GPTConfig) -> 'GPTDecoderBlock':
    return cls(config.d_model, config.d_k, config.d_v, config.n_attn_heads, config.dim_feedfwd, config.dropout_rate)

def make_causal_mask(size: int) -> T.Tensor:
  """Generate a causual attention mask from the input embeddings.
  Args:
    embeddings: BS * n_tokens * embedding_dim
  Returns:
    causal attention mask of shape (BS, n_tokens, embedding_dim)
  """
  return T.tril(T.ones(size, size))

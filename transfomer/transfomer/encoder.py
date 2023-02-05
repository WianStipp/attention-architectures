"""This module contains the encoder."""

from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from transfomer import attn, positional_encoding

MAX_SEQUENCE_LENGTH = 10000

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
    positional_encoding = self.positional_encoder(embedding)
    embedding += positional_encoding
    for block in self.encoder_blocks:
      embedding = block(embedding, attention_mask)
    return embedding

class EncoderBlock(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.multihead_attn = attn.MultiHeadAttention(d_model, d_k, d_v, d_v*n_heads, n_heads)
    self.feedfwd = nn.Linear(d_model, d_model)

  def forward(self, E: T.Tensor, attention_mask: Optional[T.Tensor] = None) -> T.Tensor:
    """
    Encoder block takes in an encoding of dims:
     (batch_size, token_len, embedding_dim)
    """
    self_attn = self.multihead_attn(Q=E, K=E, V=E)
    sublayer = F.layer_norm(self_attn + E, normalized_shape=(self.d_model, ))
    return F.layer_norm(self.feedfwd(sublayer) + sublayer, normalized_shape=(self.d_model, ))

if __name__ == "__main__":
  batch_size, n_tokens = 8, 69
  vocab_size = 50000
  d_model, d_k, d_v, n_heads, n_encoders = 512, 64, 64, 8, 6
  encoder = Encoder(vocab_size, d_model, d_k, d_v, n_heads, n_encoders)
  import tiktoken
  tokenizer = tiktoken.get_encoding("gpt2")
  encodings = tokenizer.encode_batch(['hello world!', "good day, earth!"])
  max_len = max(map(len, encodings))
  encodings = [e + [0 for _ in range(max_len - len(e))] for e in encodings]
  toks = T.Tensor(encodings).long()
  output = encoder(toks)
  print(output)
  print(output.shape)

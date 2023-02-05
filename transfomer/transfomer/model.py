"""This module contains the tranformer model."""

from typing import Optional
import dataclasses
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from transfomer import decoding, encoding

@dataclasses.dataclass
class TransfomerConfig:
  vocab_size: int
  d_model: int
  d_k: int
  d_v: int
  n_attn_heads_in_encoder: int
  n_attn_heads_in_decoder: int
  n_decoder_blocks: int
  n_encoder_blocks: int

class Transformer(nn.Module):
  def __init__(self, config: TransfomerConfig) -> None:
    super().__init__()
    self.config = config
    self.encoder = encoding.Encoder(self.config.vocab_size, self.config.d_model, \
                                    self.config.d_k, self.config.d_v, \
                                      self.config.n_attn_heads_in_encoder, self.config.n_encoder_blocks
                                      )
    self.decoder = decoding.Decoder(self.config.vocab_size, self.config.d_model, \
                                    self.config.d_k, self.config.d_v, \
                                      self.config.n_attn_heads_in_decoder, self.config.n_decoder_blocks
                                      )
    self.linear = nn.Linear(self.config.d_model, self.config.vocab_size)

  def forward(self, input_tokens: T.Tensor, decoded_tokens: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
    encoder_output = self.encoder(input_tokens, mask)
    decoder_output = self.decoder(encoder_output, decoder_output, mask)
    logits = self.linear(decoded_tokens) # (BS, vocab_size)
    return F.softmax(logits, dim=1)

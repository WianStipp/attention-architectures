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
  dim_feedfwd: int
  label_smoothing: float
  start_token: int = 0
  pad_token: int = 0
  end_token: int = 1

class Transformer(nn.Module):
  def __init__(self, config: TransfomerConfig) -> None:
    super().__init__()
    self.config = config
    self.encoder = encoding.Encoder(self.config.vocab_size, self.config.d_model, self.config.dim_feedfwd,
                                    self.config.d_k, self.config.d_v, \
                                      self.config.n_attn_heads_in_encoder, self.config.n_encoder_blocks
                                      )
    self.decoder = decoding.Decoder(self.config.vocab_size, self.config.d_model, self.config.dim_feedfwd,
                                    self.config.d_k, self.config.d_v, \
                                      self.config.n_attn_heads_in_decoder, self.config.n_decoder_blocks
                                      )
    self.decoder.lut = self.encoder.lut
    self.linear = nn.Linear(self.config.d_model, self.config.vocab_size)

  def forward(self, input_tokens: T.Tensor, target_tokens: T.Tensor, input_mask: Optional[T.Tensor] = None, target_mask: Optional[T.Tensor] = None) -> T.Tensor:
    encoder_output = self.encoder(input_tokens, input_mask)
    decoder_output = self.decoder(encoder_output, target_tokens, input_mask, target_mask) # (BS, n_toks, d_model)
    logits = self.linear(decoder_output) # (BS, n_target_toks, vocab_size)
    if target_tokens is None:
      loss = None
    else:
      # len(targets) == len(target_toks) - 1
      targets = F.pad(target_tokens[:, 1:], pad=(0,1,0,0), mode='constant', value=0)
      targets[targets == self.config.end_token] = -100 # ignore these
      targets[:,1:][targets[:,1:] == self.config.pad_token] = -100 # prediction on the end token
      loss = F.cross_entropy(T.swapaxes(logits,-2,-1), targets, label_smoothing=self.config.label_smoothing)
    return logits, loss

if __name__ == "__main__":
  import tiktoken
  tokenizer = tiktoken.get_encoding("gpt2")
  encodings = tokenizer.encode_batch(['hello world!', "good day, earth!"])
  max_len = max(map(len, encodings))
  encodings = [e + [0 for _ in range(max_len - len(e))] for e in encodings]
  toks = T.Tensor(encodings).long()
  config = TransfomerConfig(50000, 512, 8, 8, 8, 8, 6, 6, dim_feedfwd=2048)
  model = Transformer(config)
  out = model(toks, toks)
  print(out.shape)
  print(out[0])

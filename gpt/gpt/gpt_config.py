from typing import NamedTuple

class GPTConfig(NamedTuple):
  vocab_size: int
  d_model: int
  dim_feedfwd: int
  d_k: int
  d_q: int
  d_v: int
  n_attn_heads: int
  n_decoder_blocks: int
  dropout_rate: float

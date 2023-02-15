import torch as T
from torch import nn

class MultiHeadAttention(nn.Module):
  def __init__(self, n_attn_heads: int, d_model: int, d_q: int, d_k: int, d_v: int, attn_type: str) -> None:
    super().__init__()
    if attn_type == DotProductAttention.__name__:
      self.attention_heads = nn.ModuleList([DotProductAttention(d_model, d_q, d_k, d_v) for _ in range(n_attn_heads)])
    elif attn_type == DotProductAttention.__name__:
      self.attention_heads = nn.ModuleList([LocalitySensitiveHashingAttention(d_model, d_q, d_k, d_v) for _ in range(n_attn_heads)])
    else: raise ValueError(f'{attn_type=} not recognized')

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor):
    head_outputs = T.concat([a for a in self.attention_heads])


class DotProductAttention(nn.Module):
  def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int) -> None:
    super().__init__()
    self.query_proj = nn.Linear(d_model, d_q)
    self.key_proj = nn.Linear(d_model, d_k)
    self.value_proj = nn.Linear(d_model, d_v)

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
    QW = self.query_proj(Q)
    KW = self.key_proj(K)
    VW = self.value_proj(V)

class LocalitySensitiveHashingAttention(nn.Module):
  def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int) -> None:
    super().__init__()

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
    ...


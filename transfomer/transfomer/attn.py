from typing import List, Optional
import torch as T
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int, d_o:int, n_heads: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.d_o = d_o
    self.heads: List[Attention] = nn.ModuleList([Attention(d_model, d_k, d_v) for _ in range(n_heads)])
    self.output_projection = nn.Linear(n_heads * d_v, d_model)

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
    head_outputs = T.concat([head(Q, K, V, mask) for head in self.heads], dim=-1)
    return self.output_projection(head_outputs)

class Attention(nn.Module):
  def __init__(self, d_model: int, d_k: int, d_v: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.query_projection = nn.Linear(d_model, d_k)
    self.key_projection = nn.Linear(d_model, d_k)
    self.value_projection = nn.Linear(d_model, d_v)

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: Optional[T.Tensor] = None) -> T.Tensor:
    """Compute attention given the query, keys and values.
    Args:
      Q: BS * n_tokens * model_dims
      K: BS * n_tokens * model_dims
      V: BS * n_tokens * model_dims
      mask: BS * n_tokens or BS * n_tokens * n_tokens. Use the former
        for masking padded tokens and the latter for causal attention masking.
    """
    QW = self.query_projection(Q)
    KW = self.key_projection(K)
    VW = self.value_projection(V)
    scores = T.bmm(QW, T.swapaxes(KW, -1, -2)) # BS, n_toks, n_toks
    if mask and len(mask.shape) == 2: # mask for padding purposes
      mask = (1 - T.bmm(mask[:,:,None], mask[:,None,:])) * -1e9
      scores += mask
    elif mask and len(mask.shape) == 3: # mask for causual attention
      mask = mask * -1e9
      scores += mask
    return T.bmm(T.softmax(scores, dim=-1) / self.d_k ** 1/2, VW)

def make_causal_mask(size: int) -> T.Tensor:
  """Generate a causual attention mask from the input embeddings.
  Args:
    embeddings: BS * n_tokens * embedding_dim
  Returns:
    causal attention mask of shape (BS, n_tokens, embedding_dim)
  """
  return T.tril(T.ones(size, size))

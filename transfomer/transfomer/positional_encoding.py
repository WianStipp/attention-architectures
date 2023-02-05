import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAX_SEQ_LENGTH = 10000

class PositionalEncoding(nn.Module):
  """Encodes sequence positions into the transformer."""
  def __init__(self, d_model: int) -> None:
    super().__init__()
    # pos is the token position, i is the dimention
    # PE(pos, 2i) = sin(pos / MAX_SEQ_LEN ** (2*i/d_model))
    # PE(pos, 2i+1) = cos(pos / MAX_SEQ_LEN ** (2*i/d_model))
    # (MAX_SEQ, d_model) shape
    numerator = T.stack([T.arange(0,MAX_SEQ_LENGTH) for _ in range(d_model)]).T
    denominator = (MAX_SEQ_LENGTH ** ((2*T.arange(d_model))/d_model))
    pe = numerator / denominator
    pe[::2] = T.sin(pe[::2])
    pe[1::2] = T.cos(pe[1::2])
    self.register_buffer('pe', pe)

  def forward(self, encoding: T.Tensor) -> T.Tensor:
    """Position encode the input encoding.

    Args:
      encoding: a tensor of shape (batch_size, n_tokens, d_model)
    """
    return encoding + Variable(self.pe[:encoding.size(1), :], requires_grad=False)

if __name__ == "__main__":
  pe = PositionalEncoding(786)
  t = T.randn(8, 10, 786)
  output = pe(t)
  print(output.shape)

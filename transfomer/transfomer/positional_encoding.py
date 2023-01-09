import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAX_SEQ_LENGTH = 10000

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int) -> None:
    super().__init__()
    pe = T.zeros(MAX_SEQ_LENGTH, d_model)
    # half implemented!
    self.register_buffer('pe', self.pe)

  def forward(self, encoding: T.Tensor) -> T.Tensor:
    """Position encode the input encoding."""
    return encoding + Variable(self.pe[:, :encoding.size(1)], requires_grad=False)


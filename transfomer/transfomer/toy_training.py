"""Toy training script to make sure the model is learning as expected."""

from typing import List, Tuple, NamedTuple
import random
import torch as T
from torch.utils.data import Dataset

from transfomer import modeling

MAX_TRAINING_SEQ_LEN = 32
PAD_TOK = 0
START_TOK = 0
EOS_TOK = 1

class TrainingPoint(NamedTuple):
  encoder_tokens: T.Tensor
  decoder_tokens: T.Tensor
  target_token: T.Tensor

class ToyDataset(Dataset):
  def __init__(self, input_output_sequences: List[Tuple[T.Tensor, T.Tensor]]) -> None:
    self.input_output_sequences = input_output_sequences
    self._prepare_data()

  def _prepare_data(self) -> None:
    self.data = []
    for input, target in self.input_output_sequences:
      input = apply_start_end_tokens(input)
      target = apply_start_end_tokens(target)
      for i in range(1, len(target)):
        t = target[i]
        decode_toks = target[:i]
        self.data.append(TrainingPoint(input, decode_toks, t))

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index) -> TrainingPoint:
    return self.data[index]

def make_dataset(n_examples: int, max_vocab_tok: int):
  """
  Toy dataset where we encode a random sequence, and the task is to decode it backwards.
  I.e. (1,8,12,9) -> (9,12,8,1)
  """
  input_output_sequences: List[Tuple[T.Tensor, T.Tensor]] = []
  for _ in range(n_examples):
    seq_len = random.randint(1, MAX_TRAINING_SEQ_LEN)
    x = T.randint(low=1, high=max_vocab_tok+1, size=(seq_len, )).type(T.long)
    y = T.flip(x, (0, ))
    input_output_sequences.append((x,y))
  return ToyDataset(input_output_sequences)

def apply_start_end_tokens(x: T.Tensor, include_end: bool = True) -> T.Tensor:
  if include_end:
    return T.concat([T.Tensor((START_TOK, )), x, T.Tensor((EOS_TOK, ))], dim=0).type(T.long)
  return T.concat([T.Tensor((START_TOK, )), x], dim=0).type(T.long)

def main() -> None:
  config = modeling.TransfomerConfig(vocab_size=100, d_model=512, d_k=64, d_v=64, n_attn_heads_in_encoder=8,\
                                      n_attn_heads_in_decoder=8, n_decoder_blocks=6, n_encoder_blocks=6, dim_feedfwd=2048\
                                    )
  model = modeling.Transformer(config)
  dataset = make_dataset(1, max_vocab_tok=config.vocab_size)
  for d in dataset:
    print(d)

if __name__ == '__main__':
  main()

"""Toy training script to make sure the model is learning as expected."""

import tiktoken
from transfomer import modeling

def make_dataset():
  """
  Toy dataset where we encode a random sequence, and the task is to decode it backwards.
  I.e. (1,8,12,9) -> (9,12,8,1)
  """

def main() -> None:
  config = modeling.TransfomerConfig(100, 512, 64, 64, 8, 8, 6, 6)
  model = modeling.Transformer(config)

if __name__ == '__main__':
  main()

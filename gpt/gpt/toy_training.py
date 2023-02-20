"""This is a toy training script to check that the model is
training as expected."""

import random

from gpt import modeling, gpt_config

def make_training_data():
  """
  The training task is to learn the alphabet, i.e.
  given a sequence of characters, output the next letter in
  the alphabet. E.g. 'ghi' -> 'j'.
  We use the convention that after 'z', we wrap back around to 'a'.
  """
  ...

def main():
  config = gpt_config.GPTConfig(26, 512, 1024, 32, 32, 32, 4, 4, 0.1)
  model = modeling.GPT(config)
  print('number of parameters:', sum([p.numel() for p in model.parameters()]))

if __name__ == "__main__":
  main()

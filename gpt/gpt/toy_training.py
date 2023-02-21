"""This is a toy training script to check that the model is
training as expected."""

from typing import NamedTuple, Sequence
import random
import torch as T
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gpt import modeling, gpt_config

N_EPOCHS = 3
END_TOKEN = 27
PADDING_ID = 0
BATCH_SIZE = 4

class TrainingPoint(NamedTuple):
  tokens: T.Tensor
  padding_mask: T.Tensor

def make_training_data(n_datapoints: int):
  """
  The training task is to learn the alphabet,
  I.e. w -> xyz
  """
  return [_make_training_datapoint() for _ in range(n_datapoints)]

def _make_training_datapoint():
  start_point = random.randint(0, 26)
  tokens = T.Tensor(list(range(start_point, END_TOKEN + 1))).type(T.long)
  return TrainingPoint(tokens, T.ones(len(tokens)))

def collate_trainingpoints(trainingpoints: Sequence[TrainingPoint]) -> TrainingPoint:
  tokens = [t.tokens for t in trainingpoints]
  masks = [t.padding_mask for t in trainingpoints]
  max_tokens = len(max(tokens, key=len))
  batched_tokens = T.vstack([F.pad(input=t, pad=(0,max_tokens-len(t)), mode='constant', value=PADDING_ID) for t in tokens])
  batched_mask = T.vstack([F.pad(input=m, pad=(0,max_tokens-len(m)), mode='constant', value=PADDING_ID) for m in masks])
  return TrainingPoint(batched_tokens.type(T.long), batched_mask.type(T.bool))

def main():
  config = gpt_config.GPTConfig(26+2, 512, 1024, 32, 32, 32, 4, 4, 0.1)
  model = modeling.GPT(config)
  device = T.device("cuda" if T.cuda.is_available() else 'cpu')
  model.to(device)
  print('number of parameters:', sum([p.numel() for p in model.parameters()]))

  optimizer = optim.AdamW(model.parameters(), lr=1e-3)
  data = make_training_data(10000)
  dataloader = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=collate_trainingpoints)
  for _ in range(N_EPOCHS):
    for batch in dataloader:
      optimizer.zero_grad()
      batch = [b.to(device) for b in batch]
      _, loss = model(*batch)
      loss.backward()
      optimizer.step()
      print(loss)


if __name__ == "__main__":
  main()

"""Toy training script to make sure the model is learning as expected."""

from typing import List, Tuple, NamedTuple, Sequence, Optional, Iterable
import tqdm
import random
import torch as T
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transfomer import modeling, attn

MAX_TRAINING_SEQ_LEN = 32
PAD_TOK = 0
START_TOK = 0
EOS_TOK = 1

# TRAINING ARGS
BATCH_SIZE = 4

class TrainingPoint(NamedTuple):
  encoder_tokens: T.Tensor
  decoder_tokens: T.Tensor
  encoder_mask: Optional[T.Tensor]
  decoder_mask: Optional[T.Tensor]

class ToyDataset(Dataset):
  def __init__(self, input_output_sequences: List[Tuple[T.Tensor, T.Tensor]]) -> None:
    self.input_output_sequences = input_output_sequences
    self._prepare_data()

  def _prepare_data(self) -> None:
    self.data = []
    for input, target in self.input_output_sequences:
      target = apply_start_end_tokens(target)
      self.data.append(TrainingPoint(input, target, T.ones(len(input)), attn.make_causal_mask(len(target))))

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index) -> TrainingPoint:
    return self.data[index]

def collate_trainingpoints(trainingpoints: Sequence[TrainingPoint]) -> TrainingPoint:
  encoder_toks = [t.encoder_tokens for t in trainingpoints]
  encoder_mask = [t.encoder_mask for t in trainingpoints]
  decoder_toks = [t.decoder_tokens for t in trainingpoints]
  decoder_mask = [t.decoder_mask for t in trainingpoints]
  assert not any([m is None for m in encoder_mask])
  assert not any([m is None for m in decoder_mask])
  max_encoder_len = len(max(encoder_toks, key=len))
  max_decoder_len = len(max(decoder_toks, key=len))
  encoder_toks = T.vstack([F.pad(input=e, pad=(0,max_encoder_len - len(e)), mode='constant', value=PAD_TOK) for e in encoder_toks])
  decoder_toks = T.vstack([F.pad(d, (0, max_decoder_len - len(d)), 'constant', PAD_TOK) for d in decoder_toks])
  encoder_mask = T.vstack([F.pad(input=m, pad=(0,max_encoder_len - len(m)), mode='constant', value=PAD_TOK) for m in encoder_mask])
  decoder_mask = T.stack([F.pad(input=m, pad=(0,max_decoder_len - len(m),0,max_decoder_len - len(m)), mode='constant', value=PAD_TOK) for m in decoder_mask])
  return TrainingPoint(encoder_toks, decoder_toks, encoder_mask, decoder_mask)

def make_dataset(n_examples: int, max_vocab_tok: int):
  """
  Toy dataset where we encode a random sequence, and the task is to decode it backwards.
  I.e. (1,8,12,9) -> (9,12,8,1)
  """
  input_output_sequences: List[Tuple[T.Tensor, T.Tensor]] = []
  for _ in range(n_examples):
    seq_len = random.randint(1, MAX_TRAINING_SEQ_LEN)
    x = T.randint(low=2, high=max_vocab_tok, size=(seq_len, )).type(T.long)
    y = T.flip(x, (0, ))
    input_output_sequences.append((x,y))
  return ToyDataset(input_output_sequences)

def apply_start_end_tokens(x: T.Tensor, include_end: bool = True) -> T.Tensor:
  if include_end:
    return T.concat([T.Tensor((START_TOK, )), x, T.Tensor((EOS_TOK, ))], dim=0).type(T.long)
  return T.concat([T.Tensor((START_TOK, )), x], dim=0).type(T.long)

def main() -> None:
  config = modeling.TransfomerConfig(vocab_size=5, d_model=512, d_k=64, d_v=64, n_attn_heads_in_encoder=8, \
                                    n_attn_heads_in_decoder=8, n_decoder_blocks=6, n_encoder_blocks=6, dim_feedfwd=2048, \
                                    label_smoothing=0.00
                                    )
  dataset = make_dataset(10000, max_vocab_tok=config.vocab_size)
  # dataset = [dataset[0] for d in dataset]
  dataloader: Iterable[TrainingPoint] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_trainingpoints)
  device = T.device('cuda' if T.cuda.is_available() else 'cpu')
  model = modeling.Transformer(config)
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  model.train()
  running_loss = []
  for _ in range(5):
    for batch in tqdm.tqdm(dataloader, total=len(dataset) // BATCH_SIZE):
      batch = (b.to(device) for b in batch)
      optimizer.zero_grad()
      _, loss = model(*batch)
      loss.backward()
      running_loss.append(loss)
      if len(running_loss) >= 50:
        print(sum(running_loss[-50:]) / 50)
      optimizer.step()

if __name__ == '__main__':
  main()

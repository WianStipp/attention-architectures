# Reformer

Implementing the paper:

Reformer: The Efficient Transformer, Kitaev et al. https://arxiv.org/pdf/2001.04451.pdf

There are a couple of main differences to the original Vaswani et al. architecture, the main one being to replace dot-product attention with attention that uses locality-sensitive hashing, reducing complexity to O(L*logL), where L is the sequence length.

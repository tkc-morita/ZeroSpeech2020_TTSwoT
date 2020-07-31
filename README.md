# ZeroSpeech 2020 submission by Morita & Koda

This repository provides the PyTorch code used for our submission to [ZeroSpeech 2020](http://zerospeech.com/2020/) (tackling the 2019 task).

The manuscript was accepted in INTERSPEECH 2020 and will become available in its proceedings.
There is also a preprint in [arXiv](https://arxiv.org/abs/2005.05487).

Three different models were investigated in the paper:
- [`ESN/`](ESN/): The model submitted to the challenge. 
- [`ESN_F0-learn/`](ESN_F0-learn/): A modified version of `ESN` that computes the loss between its internal representation of F0 (input to the neural source-filter model) against F0 of the data.
- [`conv_encoder/`](conv_encoder/): Replacement of the ESN encoder with the CNN adopted in [Chorowski et al. (2019)](https://ieeexplore.ieee.org/document/8822475/).


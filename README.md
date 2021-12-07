# Lie Algebra Convolutional Network (L-conv) implementation  
__Paper:__ [Automatic Symmetry Discovery with Lie Algebra Convolutional Network](https://papers.nips.cc/paper/2021/file/148148d62be67e0916a833931bd32b26-Paper.pdf) _Nima Dehmamy, Robin Walters, Yanchen Liu, Dashun Wang, Rose Yu_ NeurIPS 2021  
(find updated versions on [arxiv](https://arxiv.org/abs/2109.07103)) 


## Contents
A simple implementation of the L-conv layer in PyTorch (>=1.8) can be found in `src/lconv.py`.
The L-conv layer acts similar to a graph convlutional layer (GCN), so prepare your input in a similar fashion (e.g. flatten the spatial dimensions). 
The input should have shape `(batch, channels, #nodes)` (e.g. on an image, # nodes = # pixels)  
This repository also contains code and notebooks for the experiemnts in the paper (appendix C and D) under `paper-code`. 
Most experiments in appendix D use an older (but identical) implementation in Tensoflow (>=2.1). 
Comparison with LieConv in appendix D requires the [LieConv](https://github.com/mfinzi/LieConv) packages. 

### TBA soon:
Exmaples of uses will be added soon.

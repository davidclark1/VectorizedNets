# Credit Assignment Through Broadcasting a Global Error Vector   
David G. Clark, L.F. Abbott, SueYeon Chung  

This repository contains code for training the models and generating the figures from our [NeurIPS 2021 paper](https://arxiv.org/abs/2106.04089).

## Main dependencies  
* PyTorch
* NumPy
* SciPy
* scikit-learn

## Overview of code
Our code is organized as follows:
* `vnn.py`: Custom vectorized layers. We implemented our own non-autograd code for the backward pass in vectorized networks since BP in vectorized models can be performed by backpropagating an unvectorized signal, whereas autograd backpropagates a vectorized signal by default, performing $K$ times more computations than necessary.
* `init_methods.py`: Implementation of He and ON/OFF initializations.
* `local2d.py`: Custom locally connected layer (following [Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures](https://arxiv.org/abs/1807.04587)) with an efficient backward function.
* `dfa_util.py`: Modified code from [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures](https://github.com/lightonai/dfa-scales-to-modern-deep-learning) for DFA in conventional networks.
* `vec_models.py`: Specifications of vectorized models.
* `nonvec_models.py`: Specifications of conventional models.
* `train_models.py`: Training script for running all 48 experiments in Tables 1 and 2 of the main text. 


Contact: David G. Clark <dgc2138@cumc.columbia.edu>
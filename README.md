# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
Reimplementation of the DeepAR paper(https://arxiv.org/abs/1704.04110) in PyTorch.

## Authors:
Yunkai Zhang, University of California, Santa Barbara
Qiao Jiang, Brown University


## To run:

0. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.
Download the dataset and preprocess the data:
1. python preprocess_elect.py
Start training:
2. python train.py
If you want to perform ancestral sampling,
2. python train.py --sampling
If you do not want to do normalization during evaluation,
2. python train.py --relative-metrics
Evaluate a set of saved model weights:
3. python evaluate.py
Perform hyperparameter search:
4. python search_params.py

## Results

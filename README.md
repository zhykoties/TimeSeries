# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
Reimplementation of the DeepAR paper(https://arxiv.org/abs/1704.04110) in PyTorch.

## Authors:
* **Yunkai Zhang**(<yunkai_zhang@ucsb.edu>) - *University of California, Santa Barbara* 

* **Qiao Jiang** - *Brown University*

## To run:

0. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.
1. Download the dataset and preprocess the data:
```bash
python preprocess_elect.py
```
2. Start training:
```bash
python train.py
```
If you want to perform ancestral sampling,
```bash
python train.py --sampling
```
If you do not want to do normalization during evaluation,
```bash
python train.py --relative-metrics
```
3. Evaluate a set of saved model weights:
```bash
python evaluate.py
```
4. Perform hyperparameter search:
```bash
python search_params.py
```

## Results

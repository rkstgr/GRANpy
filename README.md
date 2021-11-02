Graph Regulatory Autoencoder Network in python (GRANpy) for Gene Regulatory Network completion using scRNA-Seq datasets
============ 

## Requirements
* TensorFlow (1.0 or later) with TensorBoard -- tested with tf1.14.0
* python 2.7
* networkx
* scikit-learn
* scipy

## Run the demo (Yeast scRNA-Seq dataset)

```bash
python main.py
```

## Options

### --dataset
> Default: yeast_gasch

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

### --learning_rate
> Default: 0.00001

Initial learning rate.

### --epochs
> Default: 1000

Number of epochs to train.

### --hidden1
> Default: 64

Number of units in hidden layer 1.

### --hidden2
> Default: 48

Number of units in hidden layer 2.

### --weight_decay
> Default: 0

Weight for L2 loss on embedding matrix.

### --dropout
> Default: 0

Dropout rate (1 - keep probability).

### --early_stopping
> Default: 5

Tolerance for early stopping (# of epochs).

### --model
> Default: gcn_ae

You can choose between the following models: 
* `gcn_ae`: Graph Auto-Encoder (with GCN encoder)
* `gcn_vae`: Variational Graph Auto-Encoder (with GCN encoder)

### --features
> Default: 1

Whether to use features (1) or not (0).

### --crossvalidation
> Default: 0

Whether to use crossvalidation (1) or not (0).

### --hp_optimization
> Default: 0

Whether to start the hyperparameter optimization run (1) or not (0).

## Original paper by Kipf et. al. 2016 (Graph Auto-Encoders)

```
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

Graph Auto-Encoders (GAEs) are end-to-end trainable neural network models for unsupervised learning, clustering and link prediction on graphs. 

GAEs have successfully been used for:
* Link prediction in large-scale relational data: M. Schlichtkrull & T. N. Kipf et al., [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017),
* Matrix completion / recommendation with side information: R. Berg et al., [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017).


GAEs are based on Graph Convolutional Networks (GCNs), a recent class of models for end-to-end (semi-)supervised learning on graphs:

T. N. Kipf, M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR (2017). 

A high-level introduction is given in our blog post:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

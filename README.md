## Environment Setup
```powershell
conda create --name NeuralNetworksAndDeepLearning
conda activate NeuralNetworksAndDeepLearning
conda install mkl-service
conda install mkl
conda install -c anaconda ipykernel nbconvert
python -m ipykernel install --user --name=NNaDL
```
NNaDL = Neural Networks and Deep Learning <br>
**Switch to NNaDL kernel in Jupyter Notebook**

## Contents
#### 1. Basic matrix operations in NumPy. Visualization of weights of the neural network. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B1%5D%20visualization%20of%20weights%20of%20the%20neural%20network.ipynb)
- implementation of the sigmoid activation function
- implementation of the feed forward operation in one-layer neural network
- visualization of weights of the neural network

#### 2. Visualizations and classification based on MNIST - digits dataset. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B2%5D%20data%20visualization%20and%20classification.ipynb)
- visualizations of the distribution of MNIST digits
- Principal Component Analysis (PCA) - 2D and 3D (plotly)
- T-distributed Stochastic Neighbour Embedding (T-SNE)
- Classification using SVM
- Analysis of confusion matrix
- boolean indexing in NumPy

#### 3. Restricted Boltzmann Machine and Contrastive Divergence algorithm. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B3%5D%20Restricted%20Boltzmann%20Machine%20and%20Contrastive%20Divergence%20algorithm.ipynb)
- Restricted Boltzmann Machine
- Contrastive Divergence algorithm
- Gibbs sampling
- Stochastic gradient descent


#### 4. Evaluation and comparison of CDk and PCD algorithms. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B4%5D%20Comparison%20of%20Contrastive%20Divergence%20and%20Persistent%20Contrastive%20Divergence%20algorithms.ipynb)
- Implementation of Persistent Contrastive Divergence algorithm
- Comparision of RBM's training algorithms based on MNIST dataset

#### 5. Training RBM with Momentum and Introduction to DBNs (Deep Belief Networks). [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B5%5D%20Training%20RBM%20with%20Momentum%20and%20introduction%20to%20Deep%20Belief%20Networks.ipynb)
- RBM training with Momentum - modification of classical SGD algorithm
- DBN greedy layer-wise training
- DBN sampling

#### 6. Backpropagation algorithm. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B6%5D%20Backpropagation%20algorithm.ipynb)
- Two phases of the algorithm: forward pass and error backpropagation
- MLP training with Backpropagation and minibatched variant of SGD

#### 7. L1L2 Regularization and initialization of MLP weights. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B7%5D%20L1L2%20Regularization%20and%20DBN%20pre-training.ipynb)
- L1 and L2 regularization
- MLP pre-training using DBN
- Comparison of results of plan MLP and pre-trained MLP

#### 8. ReLU activation function and Max-Norm Regularization. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B8%5D%20ReLU%20and%20Max-Norm%20Regularization.ipynb)
- Changing Sigmoid to ReLU activation function
- Max-Norm Reguralization. Introduction of competitive weights
- Initializing MLP with DBN weights

#### 9. Dropout Regularization. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B9%5D%20Dropout%20Regularization.ipynb)

#### 10. Autoencoder. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B10%5D%20Autoencoder.ipynb)
- Autoencoder initialization with DBN weights
- Autoencoder training

#### 11. Autoencoder training with Nesterov accelerated gradient descent. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B11%5D%20Autoencoder%20training%20with%20Nesterov%20accelerated%20gradient%20descend.ipynb)

#### 12. Convolutional Neural Network. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B12%5D%20Convolutional%20Neural%20Network.ipynb)

#### 13. Negative Sampling Algorithm. [View](https://github.com/Smendowski/neural-networks-and-deep-learning/blob/main/%5B13%5D%20Negative%20Sampling%20Algorithm.ipynb)
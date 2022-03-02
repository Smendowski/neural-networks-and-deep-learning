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
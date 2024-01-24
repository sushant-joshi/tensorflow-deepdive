
# Tensorflow Bootcamp

## Objectives:

- [x] Master advanced Tensorflow features for building and training neural networks
- [] Develop an understanding of Tensorflow's distributed computing capabilities
- [] Learn how to optimize Tensorflow code for performance on GPUs and other hardware
- [] Understand best practices for training large-scale models in Tensorflow
- [] Explore advanced techniques for debugging and profiling Tensorflow code
- [] Understand the latest research developments in Tensorflow and their applications

## Outline:

#### Advanced Neural Network Architectures

- [] Introduction to advanced neural network architectures
- [] Convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers
- [] State-of-the-art models for computer vision and natural language processing (NLP)

#### Distributed Computing with Tensorflow

- [] Introduction to Tensorflow's distributed computing capabilities
- [] Setting up and configuring distributed training
- [] Synchronization techniques, data parallelism, and model parallelism

#### Tensorflow Performance Optimization

- [] Overview of Tensorflow performance optimization techniques
- [] Best practices for optimizing Tensorflow code for GPUs and other hardware
- [] Mixed-precision training, parallelism, and optimizing memory usage

#### Large-scale Model Training in Tensorflow

- [] Scaling up training to large datasets
- [] Strategies for efficient data loading, preprocessing, and augmentation
- [] Distributed training on large-scale datasets

#### Advanced Tensorflow Debugging and Profiling

- [] Overview of Tensorflow's debugging and profiling tools
- [] Advanced techniques for debugging Tensorflow code
- [] Profiling Tensorflow code for performance bottlenecks

#### Tensorflow Research Developments and Applications

- [] Latest research developments in Tensorflow
- [] Applications of Tensorflow in domains such as healthcare, robotics, and autonomous driving
- [] Future directions and trends in deep learning and Tensorflow

#### Capstone Project

- [] Apply the knowledge and skills acquired throughout the bootcamp to a practical project
- [] Develop and train an advanced deep learning model using Tensorflow
- [] Present and showcase the project to the world




```
https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/ 

1. Prepare the Data. 
    - Tensorflow provides the Dataset class that you can extend and customize to load your dataset.
    - Tensorflow provides the DataLoader class to navigate a Dataset instance during the training and evaluation of your model.
2. Define the Model.
    - The idiom for defining a model in Tensorflow involves defining a class that extends the Module class.
3. Train the Model.
    - The training process requires that you define a loss function and an optimization algorithm.
        - loss functions - Cross Entropy Loss, BCE, MeanSquaredLoss
        - optimization techniques - SGD, Adam
    - Each Epoch has the following steps 
        - Clearing the last error gradient.
        - A forward pass of the input through the model.
        - Calculating the loss for the model output.
        - Backpropagating the error through the model.
        - Update the model in an effort to reduce loss.
4. Evaluate the Model.
    - Compare predictions vs target data and compute eval metrics 
5. Make Predictions on unseen data
    - This requires that you wrap the data in a Tensorflow Tensor data structure. A Tensor is just the Tensorflow version of a NumPy array for holding data. It also allows you to perform the automatic differentiation tasks in the model graph, like calling backward() when training the model. The prediction too will be a Tensor, although you can retrieve the NumPy array by detaching the Tensor from the automatic differentiation graph and calling the NumPy function.
```

```
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
```
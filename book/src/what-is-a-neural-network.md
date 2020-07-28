# What is a Neural Network?

Often referred to as "Deep Learning", Neural Networks are composed of one or more layers of "Neurons", that map some input x to some output y. Typically this is 'y = a(x*w + b)', where y, x, and w are matrices, b is a vector, and a is an "Activation Function". An activation function is nonlinear, this helps the model approximate the dataset. 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png?raw=true)\
*[Artificial_neural_network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg)*

In the above image each vertical column of neurons represents a single 'y = a(x*w + b)', where there is a weight entry in w for each input entry in x. A neuron can be thought of a weighted some of the inputs plus a bias and mapped with an activation. The actual implementation uses matrices, with a batch of inputs, which is more efficient. A Neural Network is a supervised algorithm, it is trained by providing it with a target (ie the correct output). A loss function computes the error of the model output agaist the target. The goal of the model is to minimize this loss, which corresponds to approximating the dataset. In order to iteratively improve the prediction y, the parameters (w and b) are updated. This is done via "Gradient Descent". All functions used to compute the loss are differentiable with respect to either their inputs and or their parameters. These gradients can be propagated backward, from the loss to the parameters. The parameters and then updated with some form of 'w -= lr * dw', where lr is the "learning_rate", and dw is the gradient of the weight. The same is applied to the bias. 

Prior to training, the weights of the model are initialized, perhaps from a normal distribution, while the biases are typically initialized as zeros. Input data is batched and fed into the model as a matrix, or a higher dimensional tensor (in the case of images). Larger batches are more efficient to compute than individual samples. The "Forward Pass" is computing the prediction of the model y and the loss. The "Backward Pass" is the computation of the gradients of the parameters. For each forward operation, there are one or more backward ops that compute the gradients of the inputs / parameters given the gradient of the output. This is the backward graph. Typically the backward ops are executed in First In Last Out order, ie in reverse. The forward, backward, and update steps are repeated for all of the training batches, this is an epoch. Training typically requires many epochs for model to converge to an accurate solution. 





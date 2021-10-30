# Neural Network MNIST
This example shows how to train a Neural Network on the MNIST dataset.

# Usage
```
// Linear model
cargo run --release -- --linear
// Convolution Neural Network
cargo run --release -- --cnn
// The LeNet5 Network
cargo run --release -- --lenet5
```

# Saving and Loading
Use the `--save` flag to save progress at each epoch that can be resumed later.
```
cargo run --release -- --lenet5 --save
```

# Final Notes
If you have any issues with this example please create an issue at https://github.com/charles-r-earp/autograph.

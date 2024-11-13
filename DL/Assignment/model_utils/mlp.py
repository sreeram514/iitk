import numpy as np
from tqdm import tqdm


class ManualMLP:
    
    def __init__(self, input_size, hidden_layers, output_size, learning_rate, epochs, activation_function="relu",
                 initialisation_function="He"):
        """
        Initializes the forward pass of an MLP with the given input architecture.

        Parameters:
        ----------
        input_size : int
            Number of input features (784 for flattened 28x28 images)

        hidden_layers : list of int
            List specifying the number of neurons in each hidden layer (e.g., [128, 64])
        
        output_size : int
            Number of output classes for classification (10 for Fashion MNIST unique class labels)

        activation_function : string  Supported relu, sigmoid, tanh
        initialisation_function : string Supported random, Xavier, He
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.initialisation_function = initialisation_function
        self.model_name = (f"mlp__layers__{'_'.join(map(str, hidden_layers))}__lr_{learning_rate}__epoch_{epochs}__"
                           f"activation_{activation_function}__initiation_{initialisation_function}")
        # Define the layer sizes, including input and output layers
        layer_sizes = [input_size] + hidden_layers + [output_size]
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            if initialisation_function == "He":
                weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])  # He initialization
            elif initialisation_function == "Xavier":
                limit = np.sqrt(6.0/(layer_sizes[i] + layer_sizes[i + 1]))
                weight = np.random.uniform(low=-limit, high=limit, size=(layer_sizes[i], layer_sizes[i + 1]))
            else:
                weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
        self.activation_function = self.__getattribute__(activation_function)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x):
        """
        Applies the ReLU activation function element-wise to the input array.

        Parameters:
        ----------
        x : numpy.ndarray
            The input array to apply ReLU on.

        Returns:
        -------
        numpy.ndarray
            The output array where each element is the ReLU of the input.
        """
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        """
        Applies the softmax function to the input array row-wise (for each sample in the batch).

        Parameters:
        ----------
        x : numpy.ndarray
            The input array (logits) to apply softmax on, typically the output layer.

        Returns:
        -------
        numpy.ndarray
            The softmax-transformed array with values in the range [0, 1] and rows summing to 1.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement by subtracting max
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        """
        Computes the cross-entropy loss between predictions and true labels.

        Parameters:
        ----------
        y_pred : numpy.ndarray
            The predicted probabilities from the model (softmax output).

        y_true : numpy.ndarray
            The true labels, one-hot encoded.

        Returns:
        -------
        float
            The computed cross-entropy loss value.
        """
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))  # Adding a small value to avoid inf in log(0)

    @staticmethod
    def dropout(x, dropout_rate):
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
        return x * mask / (1 - dropout_rate)

    def forward(self, x):
        """
        Performs the forward pass through the MLP.

        Parameters:
        ----------
        x : numpy.ndarray
            The input data (e.g., a batch of flattened images) with shape (batch_size, input_size).

        Returns:
        -------
        probabilities : numpy.ndarray
            The final output probabilities of the network with shape (batch_size, output_size).
        
        activations : list of numpy.ndarray
            List of activations for each layer, useful for backward propagation.
        
        z_values : list of numpy.ndarray
            List of pre-activation values (z = Wx + b) for each layer, useful for backward propagation.
        """

        activations = []
        z_values = []

        # Forward pass through each layer except the output
        np.dot(x, self.weights[0])
        for i in range(len(self.weights) - 1):
            z = np.dot(x, self.weights[i]) + self.biases[i]  # linear-transformation
            z_values.append(z)
            x = self.activation_function(z)  # Apply activation and overwrite x so next layer gets x in correct shape to multiply with its weights
            # dropped_x = self.dropout(x, dropout_rate=0.5)
            activations.append(x)

        # Output layer (with softmax for probabilities)
        z = np.dot(x, self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        activations.append(z)
        
        # Apply softmax to output logits
        probabilities = self.softmax(z)
        return probabilities, activations, z_values

    def backward(self, x, y_true, probabilities, activations):
        """
        Performs the backward pass through the MLP.

        Parameters:
        ----------
        x : numpy.ndarray
            The input data (batch of flattened images).

        y_true : numpy.ndarray
            The true labels, one-hot encoded.

        Returns:
        -------
        gradients : dict
            Dictionary containing gradients of weights and biases.
        """
        # Number of samples
        m = x.shape[0]

        # Initialize gradients
        gradients = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        # Output layer gradients
        d_loss = probabilities - y_true
        gradients['weights'][-1] = np.dot(activations[-2].T, d_loss) / m
        gradients['biases'][-1] = np.sum(d_loss, axis=0, keepdims=True) / m

        # Back propagate through the hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            d_relu = np.dot(d_loss, self.weights[i + 1].T)                      
            d_relu[activations[i] <= 0] = 0

            # Use `x` for the input layer; otherwise, use `activations[i - 1]`
            if i == 0:
                gradients['weights'][i] = np.dot(x.T, d_relu) / m
            else:
                gradients['weights'][i] = np.dot(activations[i - 1].T, d_relu) / m

            gradients['biases'][i] = np.sum(d_relu, axis=0, keepdims=True) / m
            d_loss = d_relu
        return gradients

    def update_weights(self, gradients):
        """
        Updates the weights and biases of the MLP using the calculated gradients and the learning rate.
        """
        # Loop through each layer to update weights and biases
        for i in range(len(self.weights)):
            # Update weights and biases with gradient descent rule
            self.weights[i] -= self.learning_rate * gradients['weights'][i]
            self.biases[i] -= self.learning_rate * gradients['biases'][i]

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Trains the MLP on the given data for a specified number of epochs.
        """
        training_loss = []
        validation_loss = []
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in tqdm(range(self.epochs)):
            
            # Forward pass
            probabilities, activations, z_values = self.forward(x_train)
            
            # Compute loss
            train_pass_loss = self.cross_entropy_loss(probabilities, y_train)
            training_loss.append(train_pass_loss)

            # Backward pass
            gradients = self.backward(x_train, y_train, probabilities, activations)

            # Update weights
            self.update_weights(gradients)

            verbose_statement = f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_pass_loss:.8f}"
            
            # Compute validation loss
            if x_val is not None and y_val is not None:
                val_probabilities, _, _ = self.forward(x_val)  # Forward pass on validation set
                val_loss = self.cross_entropy_loss(val_probabilities, y_val)  # Calculate validation loss
                validation_loss.append(val_loss)
                verbose_statement += f" | Validation Loss: {val_loss:.8f}"
            print(verbose_statement)

        return training_loss, validation_loss

    def save_model_state(self):
        """
        Saves the current weights and biases in a dictionary.

        Returns:
        -------
        dict
            A dictionary with the weights and biases of the model.
        """
        return {'weights': self.weights, 'biases': self.biases,
                'activation_function': self.activation_function.__name__,
                "model_name": self.model_name}

    def load_model_state(self, state):
        """
        Loads weights and biases from a saved state into the model.

        Parameters:
        ----------
        state : dict
            A dictionary containing 'weights' and 'biases' to load into the model.
        """
        self.weights = state['weights']
        self.biases = state['biases']

    def predict(self, input_x):
        probabilities, _, _ = self.forward(x=input_x)
        return probabilities

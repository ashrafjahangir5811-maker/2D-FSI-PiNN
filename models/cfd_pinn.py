import tensorflow as tf

class CFDPINN(tf.keras.Model):
    def __init__(self, num_hidden_layers=4, neurons_per_layer=32):
        super(CFDPINN, self).__init__()
        
        # 1. The Activation Function (CRITICAL)
        # We use 'tanh' instead of 'relu'. 
        # ReLU's second derivative is zero, which will instantly break our Navier-Stokes physics loss.
        self.activation = tf.keras.activations.tanh 
        
        # 2. Build the Hidden Layers
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                tf.keras.layers.Dense(neurons_per_layer, activation=self.activation)
            )
            
        # 3. The Output Layer
        # 3 neurons for predicting (u, v, p). 
        # No activation function here (linear) because pressure and velocity can be negative or positive.
        self.final_layer = tf.keras.layers.Dense(3, activation=None)

    def call(self, inputs):
        """
        The forward pass of the network.
        Inputs: A tensor of shape (batch_size, 4) containing [X, Y, Angle, Mach] coordinates.
        Outputs: A tensor of shape (batch_size, 3) containing [u, v, p].
        """
        x = inputs
        
        # Pass the coordinates through the hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            
        # Generate the final fluid property predictions
        predictions = self.final_layer(x)
        
        return predictions

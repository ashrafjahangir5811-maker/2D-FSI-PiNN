import tensorflow as tf

class FEAPINN(tf.keras.Model):
    def __init__(self, num_hidden_layers=4, neurons_per_layer=32):
        super(FEAPINN, self).__init__()
        
        self.activation = tf.keras.activations.tanh 
        
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                tf.keras.layers.Dense(neurons_per_layer, activation=self.activation)
            )
            
        # Output: [dx, dy, sigma_xx, sigma_yy, tau_xy]
        self.final_layer = tf.keras.layers.Dense(5, activation=None)

    def call(self, inputs):
        """
        Inputs: A tensor of shape (batch_size, 2) containing [X, Y] coordinates.
        Outputs: A tensor of shape (batch_size, 5) containing [dx, dy, s_xx, s_yy, t_xy].
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            
        predictions = self.final_layer(x)
        return predictions
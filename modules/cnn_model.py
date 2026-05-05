import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

class NeuralRingerCNN:
    def __init__(self, input_shape=(50, 1), filters_1=4, filters_2=8,
                 kernel_size=2, hidden_units=16, learning_rate=0.001):
        """
        1D Convolutional Neural Network module based on ATLAS v10/v14 strategy.
        Configurable parameters for easy hyperparameter tuning.
        """
        self.input_shape = input_shape
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Builds the 1D CNN architecture for FPGA deployment."""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # Representation Layers (Spatial extraction of the 50 rings)
            Conv1D(filters=self.filters_1, kernel_size=self.kernel_size,
                   activation='relu', padding='valid', strides=1),
            Conv1D(filters=self.filters_2, kernel_size=self.kernel_size,
                   activation='relu', padding='valid', strides=1),
            
            # Flattening to feed the fully connected MLP
            Flatten(),
            
            # Discrimination Layers
            Dense(units=self.hidden_units, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ], name="NeuralRinger_CNN1D")

        # Compilation using ADAM optimizer and Binary Crossentropy [v10 strategy]
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_model(self):
        """Returns the compiled Keras model."""
        return self.model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

def build_v10_cnn(input_length, learning_rate=0.001):
  
    model = Sequential([
        Input(shape=(input_length, 1)),
        
        Conv1D(filters=4, kernel_size=2, activation='relu'),
        Conv1D(filters=8, kernel_size=2, activation='relu'),
        
        Flatten(),
        
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model
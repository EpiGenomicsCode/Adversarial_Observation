
def load_MNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_complex_MNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        # First block: Convolution + Pooling
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second block: Deeper convolution + Pooling
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third block: Deeper convolution + Pooling
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Fourth block: Additional convolutions to add complexity
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Global Average Pooling instead of Flatten for better generalization
        GlobalAveragePooling2D(),

        # Fully connected layers
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.4),

        Dense(128, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(10, activation='softmax')
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def load_AudioMNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Conv1D, MaxPooling1D

    
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(16000, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_complex_AudioMNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # First block: Convolution + Pooling
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(16000, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Second block: Deeper convolution + Pooling
        Conv1D(128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Third block: Even deeper convolution + Pooling
        Conv1D(256, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Fourth block: Additional convolution for higher complexity
        Conv1D(512, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Flatten and fully connected layers
        Flatten(),
        
        # Fully connected layer with more units
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dropout(0.4),

        Dense(256, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(10, activation='softmax')
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_MNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
     # Create a new Keras model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Summarize the model
    model.summary()
    return model

def load_complex_MNIST_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam

    # Create a new Keras model
    model = Sequential([
        Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),    
        Conv2D(64, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size = 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(128, kernel_size = 4, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Summarize the model
    model.summary()
    return model

def load_CIFAR_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam

    """
    Creates a CNN model for CIFAR-10 dataset.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Summarize the model
    model.summary()
    return model

def load_CIFAR_model(model_path=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPool2D
    from tensorflow.keras.optimizers import Adam
    

    """
    Creates a CNN model for CIFAR-10 dataset.
    """
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Summarize the model
    model.summary()
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

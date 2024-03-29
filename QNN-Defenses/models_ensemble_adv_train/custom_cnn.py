import tensorflow as tf
from tensorflow.keras import layers


def custom_cnn(input_shape, num_classes):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input)
    x = layers.Activation(activation='relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Activation(activation='relu')(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation(activation='relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(units=128)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Dense(units=32)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(units=num_classes)(x)
    output = layers.Activation(activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    
    return model
import tensorflow as tf
import dataset_loader as ds
from custom_cnn import custom_cnn
import os


def augment_cifar_dataset(x_train, y_train):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=5,
        height_shift_range=5,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1
    )
    new_ds = datagen.fit(x_train)
    new_ds = datagen.flow(x_train, y_train, batch_size=32)

    return new_ds


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    x_train_f32, y_train, x_test_f32, y_test = ds.get_cifar10_full_ds_f32()
    train_gen = augment_cifar_dataset(x_train_f32, y_train)
    num_classes = 10

    cnn = custom_cnn(input_shape=(32,32,3), num_classes=num_classes)
    cnn.layers[-1].activation = tf.keras.activations.linear

    x = cnn.layers[-1].output
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=cnn.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable=True    

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, batch_size=32, epochs=300, validation_data=(x_test_f32, y_test), workers=0, use_multiprocessing=True,
                callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath='models_ensemble_adv_train/cifar_tmp/cifar_cnn.h5', save_best_only=True,
                        monitor='val_accuracy', mode='max', save_weights_only=False, save_freq='epoch', verbose=1)]
            )
    model.evaluate(x_test_f32, y_test, workers=0, use_multiprocessing=True)
    model.save("models_ensemble_adv_train/cifar_cnn.h5")


if __name__ == '__main__':
    main()
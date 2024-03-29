import tensorflow as tf
import dataset_loader as ds


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
    x_train_f32, y_train, x_test_f32, y_test = ds.get_cifar10_full_ds_f32()
    train_gen = augment_cifar_dataset(x_train_f32, y_train)
    num_classes = 10

    vgg = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=(32, 32, 3),
        pooling='max', weights='imagenet'
    )

    x = vgg.layers[-1].output
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=vgg.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable=True    

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, batch_size=32, epochs=50, validation_data=(x_test_f32, y_test), workers=48, use_multiprocessing=True)
    model.evaluate(x_test_f32, y_test, workers=48, use_multiprocessing=True)
    model.save("cifar_vgg16.h5")


if __name__ == '__main__':
    main()

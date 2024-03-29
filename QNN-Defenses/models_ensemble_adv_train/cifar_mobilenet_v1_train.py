import tensorflow as tf
import dataset_loader as ds
from mobilenet_v1 import mobilenet_v1


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

    model = mobilenet_v1(input_shape=(32,32,3), num_classes=num_classes)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, batch_size=32, epochs=50, validation_data=(x_test_f32, y_test), workers=48, use_multiprocessing=True,
                callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath='models_ensemble_adv_train/cifar_tmp/cifar_mobilenet_v1.h5', save_best_only=True,
                        monitor='val_accuracy', mode='max', save_weights_only=False, save_freq='epoch', verbose=1)]
            )
    model.evaluate(x=x_test_f32, y=y_test, workers=48, use_multiprocessing=True)
    model.save("cifar_mobilenet_v1.h5")


if __name__ == '__main__':
    main()
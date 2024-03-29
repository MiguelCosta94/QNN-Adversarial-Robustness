import tensorflow as tf
import dataset_loader as ds
from custom_cnn import custom_cnn


def main():
    train_gen, test_gen = ds.get_vww_full_ds_f32(batch_size=32)
    num_classes = 2

    cnn = custom_cnn(input_shape=(96,96,3), num_classes=num_classes)
    cnn.layers[-1].activation = tf.keras.activations.linear

    x = cnn.layers[-1].output
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=cnn.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable=True    

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, batch_size=32, epochs=100, validation_data=test_gen, workers=48, use_multiprocessing=True,
                callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath='models_ensemble_adv_train/vww_tmp/vww_cnn.h5', save_best_only=True,
                        monitor='val_accuracy', mode='max', save_weights_only=False, save_freq='epoch', verbose=1)])
    model.evaluate(test_gen, workers=48, use_multiprocessing=True)
    model.save("models_ensemble_adv_train/vww_cnn.h5")


if __name__ == '__main__':
    main()

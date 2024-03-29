import tensorflow as tf
import dataset_loader as ds
from resnet_v1_eembc import resnet_v1_eembc


def main():
    train_gen, test_gen = ds.get_vww_full_ds_f32(batch_size=32)
    num_classes = 2

    model = resnet_v1_eembc(input_shape=(96,96,3), num_classes=num_classes)   
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, batch_size=32, epochs=100, validation_data=test_gen, workers=48, use_multiprocessing=True,
                callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath='models_ensemble_adv_train/vww_tmp/vww_resnet_eembc.h5', save_best_only=True,
                        monitor='val_accuracy', mode='max', save_weights_only=False, save_freq='epoch', verbose=1)]
            )
    model.evaluate(test_gen, workers=48, use_multiprocessing=True)
    model.save("models_ensemble_adv_train/vww_resnet_eembc.h5")
    

if __name__ == '__main__':
    main()

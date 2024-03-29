import tensorflow as tf
import dataset_loader as ds


def main():
    train_gen, val_gen, test_gen = ds.get_coffee_full_ds_f32(batch_size=32)
    ann_f32 = tf.keras.models.load_model('models_ensemble_adv_train/coffee_resnet_eembc.h5')
    ann_f32.evaluate(test_gen)

    ann_f32 = tf.keras.models.load_model('models_ensemble_adv_train/coffee_mobilenet_v1.h5')
    ann_f32.evaluate(test_gen)


if __name__ == '__main__':
    main()

import tensorflow as tf
import dataset_loader as ds


def main():
    x_train_f32, y_train, x_test_f32, y_test = ds.get_cifar10_full_ds_f32()
    ann_f32 = tf.keras.models.load_model('models_ensemble_adv_train/cifar_cnn.h5')
    ann_f32.evaluate(x_test_f32, y_test)

    ann_f32_1 = tf.keras.models.load_model('models_ensemble_adv_train/cifar_vgg16.h5')
    ann_f32_1.evaluate(x_test_f32, y_test)


if __name__ == '__main__':
    main()

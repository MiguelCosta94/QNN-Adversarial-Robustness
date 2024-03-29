import os
import pickle
import numpy as np
import tensorflow as tf


def get_cifar10_test_ds_f32(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/cifar-10-batches-py')

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar_test_data = np.array(cifar_test_data / 255).astype(np.float32)
    cifar_test_labels = tf.one_hot(cifar_test_labels, 10).numpy()

    return cifar_test_data, cifar_test_labels

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def get_vww_test_ds_f32():
    data_test = []
    labels_test = []
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/vw_coco2014_96')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(96, 96),
        batch_size=1,
        subset='validation',
        color_mode='rgb',
        shuffle=False)

    batch_index = 0
    while batch_index < val_generator.n:
        data = val_generator.next()
        data_test.append(data[0][0])
        labels_test.append(np.argmax(data[1][0]))
        batch_index = batch_index + 1

    data_test = np.array(data_test)
    labels_test = tf.one_hot(labels_test, 2).numpy()

    return data_test, labels_test


def get_coffee_test_ds_f32():
    data_test = []
    labels_test = []

    test_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/test/')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_gen = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')

    batch_index = 0
    while batch_index < test_gen.n:
        data = test_gen.next()
        img = data[0][0] / 255
        data_test.append(img)
        labels_test.append(np.argmax(data[1][0]))
        batch_index = batch_index + 1

    data_test = np.array(data_test).astype(np.float32)
    labels_test = np.array(labels_test)
    labels_test = tf.one_hot(labels_test, 4).numpy()

    return data_test, labels_test
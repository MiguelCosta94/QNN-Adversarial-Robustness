import os
import pickle
import numpy as np
import tensorflow as tf


def get_cifar10_full_ds_f32(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/cifar-10-batches-py')
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar_train_data = cifar_train_data / 255
    cifar_test_data = cifar_test_data / 255


    return cifar_train_data, tf.one_hot(cifar_train_labels, 10).numpy(), \
            cifar_test_data, tf.one_hot(cifar_test_labels, 10).numpy()

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def get_vww_full_ds_f32(batch_size):
    data_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/vw_coco2014_96')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
      data_dir,
      target_size=(96, 96),
      batch_size=batch_size,
      subset='training',
      color_mode='rgb')

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(96, 96),
        batch_size=batch_size,
        subset='validation',
        color_mode='rgb',
        shuffle=False)

    return train_generator, val_generator


def get_coffee_full_ds_f32(batch_size):
    train_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/train/')
    test_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/test/')
    val_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/val/')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, color_mode='rgb')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=batch_size, color_mode='rgb')
    test_gen = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, color_mode='rgb')

    return train_gen, val_gen, test_gen

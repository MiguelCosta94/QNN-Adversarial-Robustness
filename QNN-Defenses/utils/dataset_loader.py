import os
import pickle
import numpy as np
import tensorflow as tf


def get_cifar10_full_ds_f32(negatives=False, augmentation=False):
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
    cifar_train_data = cifar_train_data.astype(np.float32)
    cifar_test_data = cifar_test_data.astype(np.float32)
    cifar_train_labels = tf.one_hot(cifar_train_labels, 10).numpy()
    cifar_test_labels = tf.one_hot(cifar_test_labels, 10).numpy()

    if augmentation==True:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
        datagen.fit(cifar_train_data)
        datagen = datagen.flow(cifar_train_data, cifar_train_labels, batch_size=1)
        cifar_train_data, cifar_train_labels = ds_gen_to_numpy(datagen, 10)
    
    return cifar_train_data, cifar_train_labels, cifar_test_data, cifar_test_labels


def get_cifar10_train_ds_f32(negatives=False, augmentation=False):
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

    cifar_train_data = cifar_train_data / 255
    cifar_train_data = cifar_train_data.astype(np.float32)
    cifar_train_labels = tf.one_hot(cifar_train_labels, 10).numpy()

    if augmentation==True:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
        datagen.fit(cifar_train_data)
        datagen = datagen.flow(cifar_train_data, cifar_train_labels, batch_size=1)
        cifar_train_data, cifar_train_labels = ds_gen_to_numpy(datagen, 10)
    
    return cifar_train_data, cifar_train_labels


def get_vww_full_ds_f32():
    data_train, data_test, labels_train, labels_test = [], [], [], []
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
      batch_size=1,
      subset='training',
      color_mode='rgb')

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(96, 96),
        batch_size=1,
        subset='validation',
        color_mode='rgb',
        shuffle=False)
    
    data_train, labels_train = ds_gen_to_numpy(train_generator, 2)
    data_test, labels_test = ds_gen_to_numpy(val_generator, 2)

    data_train = data_train.astype(np.float32)
    data_test = data_test.astype(np.float32)

    return data_train, labels_train, data_test, labels_test


def get_vww_train_ds_f32():
    data_train, labels_train = [], []
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
      batch_size=1,
      subset='training',
      color_mode='rgb')
    
    data_train, labels_train = ds_gen_to_numpy(train_generator, 2)
    data_train = data_train.astype(np.float32)

    return data_train, labels_train


def get_coffee_full_ds_f32():
    train_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/train/')
    test_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/test/')
    #val_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/val/')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')
    train_data, train_labels = ds_gen_to_numpy(train_gen, 4)
    train_data = train_data.astype(np.float32)

    #val_gen = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')
    #val_data, val_labels = ds_gen_to_numpy(val_gen)

    test_gen = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')
    test_data, test_labels = ds_gen_to_numpy(test_gen, 4)
    test_data = test_data.astype(np.float32)

    return train_data, train_labels, test_data, test_labels


def get_coffee_train_ds_f32():
    train_dir = os.path.join(os.path.dirname(__file__), '../../Datasets/new_coffee_dataset/train/')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=1, color_mode='rgb')
    train_data, train_labels = ds_gen_to_numpy(train_gen, 4)
    train_data = train_data.astype(np.float32)

    return train_data, train_labels


def ds_gen_to_numpy(ds_generator, num_classes):
    dataset = []
    labels = []

    batch_index = 0
    while batch_index < ds_generator.n:
        data = ds_generator.next()
        img = data[0][0]
        dataset.append(img)
        labels.append(np.argmax(data[1][0]))
        batch_index = batch_index + 1

    dataset = np.array(dataset)
    labels = np.array(labels)
    labels = tf.one_hot(labels, num_classes).numpy()

    return dataset, labels


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
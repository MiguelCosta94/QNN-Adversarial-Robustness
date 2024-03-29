import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import PixelCNN, bits_per_dim_loss
from utils import PlotSamplesCallback
import tensorflow_datasets as tfds


tfk = tf.keras
tfkl = tf.keras.layers
AUTOTUNE = tf.data.experimental.AUTOTUNE


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

    #cifar_train_data = np.reshape(cifar_train_data, newshape=(-1, 64, 32, 32, 3))
    #cifar_train_labels = np.reshape(cifar_train_labels, newshape=(-1, 64))
    #cifar_test_data = np.reshape(cifar_test_data, newshape=(-1, 64, 32, 32, 3))
    #cifar_test_labels = np.reshape(cifar_test_labels, newshape=(-1, 64))

    train_ds = tf.data.Dataset.from_tensor_slices((cifar_train_data.astype(np.float32),
                                                    cifar_train_data.astype(np.float32)))
    test_ds = tf.data.Dataset.from_tensor_slices((cifar_test_data.astype(np.float32),
                                                    cifar_test_data.astype(np.float32)))

    return train_ds, test_ds


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# Parsing parameters
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=75, help='Number of training epochs')
parser.add_argument('-b', '--batch', type=int, default=32, help='Training batch size')
parser.add_argument('-bf', '--buffer', type=int, default=1024, help='Buffer size for shiffling')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='Dataset: cifar10 or mnist')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-dc', '--lr_decay', type=float, default=0.999995, help='Learning rate decay')

parser.add_argument('-hd', '--hidden_dim', type=int, default=64, help='Hidden dimension per channel')
parser.add_argument('-n', '--n_res', type=int, default=4, help='Number of res blocks')

args = parser.parse_args()


# PixelCNN training requires target = input
def duplicate(element):
    return element, element

# Training parameters
EPOCHS = args.epochs
BATCH_SIZE = args.batch
BUFFER_SIZE = args.buffer  # for shuffling
TRAIN_SAMPLES = 50000

# Load dataset
train_ds, test_ds = get_cifar10_full_ds_f32()

train_ds = (train_ds.shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

test_ds = (test_ds.batch(BATCH_SIZE)
                   .prefetch(AUTOTUNE))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_ds = train_ds.with_options(options)  # use this as input for your model
test_ds = test_ds.with_options(options)  # use this as input for your model

# Define model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = PixelCNN(
        hidden_dim=args.hidden_dim,
        n_res=args.n_res
    )
    model.compile(optimizer='adam', loss=bits_per_dim_loss)

# Learning rate scheduler
steps_per_epochs = TRAIN_SAMPLES // args.batch
decay_per_epoch = args.lr_decay ** steps_per_epochs
schedule = tfk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_rate=decay_per_epoch,
    decay_steps=1
)

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'pixelcnn', time)
tensorboard_clbk = tfk.callbacks.TensorBoard(log_dir=log_dir)
sample_clbk = PlotSamplesCallback(logdir=log_dir)
scheduler_clbk = tfk.callbacks.LearningRateScheduler(schedule)
callbacks = [tensorboard_clbk, sample_clbk, scheduler_clbk]

# Fit
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks, use_multiprocessing=True)
model.summary()
model.save("px_cnn/pixel_cnn_cifar")
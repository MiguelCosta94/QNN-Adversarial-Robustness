import os
import tensorflow as tf
import numpy as np
import glob
from argparse import ArgumentParser
import utils.backend as b


def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10, vww, or coffee")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes in target models")
    parser.add_argument("--logs_dir", type=str, default=None, help="path to the files containing the logs of the attacks")
    parser.add_argument("--qnn_int16", type=str, default=None, help="path to the int-16 QNN")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'num_classes', 'logs_dir', 'qnn_int16']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")
        
    run_cfgs = vars(cfgs)

    return run_cfgs


def dequantize(data):
    data_f32 = data * (1/32767)
    data_f32 = data_f32.astype(np.float32)
    data_f32 = np.minimum(data_f32, 1)
    data_f32 = np.maximum(data_f32, -1)

    return data_f32


def main():
    # Step 1: Load configs
    cfgs = load_configs()

    if cfgs['dataset_id'] == 'cifar10':
        input_shape = (-1,32,32,3)
    elif cfgs['dataset_id'] == 'vww':
        input_shape = (-1,96,96,3)
    elif cfgs['dataset_id'] == 'coffee':
        input_shape = (-1,224,224,3)

    # Step 2: Load ANNs/QNNs
    qnn_int16 = b.get_ml_quant_model(cfgs['qnn_int16'])

    # Step 3: Load test dataset and evaluate accuracy
    test_data_file = cfgs['logs_dir'] + '/x_test_int16.bin'
    test_labels_file = cfgs['logs_dir'] + '/labels.bin'
    test_data_file = os.path.join(os.path.dirname(__file__), test_data_file)
    test_labels_file = os.path.join(os.path.dirname(__file__), test_labels_file)

    x_test_int16 = np.fromfile(test_data_file, dtype=np.int16)
    y_test = np.fromfile(test_labels_file, dtype=np.int8)
    x_test_int16 = np.reshape(x_test_int16, input_shape)
    y_test = tf.one_hot(y_test, cfgs['num_classes']).numpy()

    accuracy = b.get_accuracy_quant_model(qnn_int16, x_test_int16, y_test)
    print("Int16 -> ACC: " + str(accuracy))
    x_test_float = dequantize(x_test_int16)

    # Step 4: Load adversarial dataset and evaluate accuracy
    for x_adv_file in sorted(glob.glob(cfgs['logs_dir'] + '/adv_int16*.bin')):
        print("##############")
        print(str(x_adv_file))
        x_adv_int16 = np.fromfile(x_adv_file, dtype=np.int16)
        x_adv_int16 = np.reshape(x_adv_int16, input_shape)

        accuracy = b.get_accuracy_quant_model(qnn_int16, x_adv_int16, y_test)
        print("Int16 -> ACC: " + str(accuracy))

        x_adv_float = dequantize(x_adv_int16)

        l0_max, l0_avg = b.get_l0_norm(x_test_float, x_adv_float)
        l1_max, l1_avg = b.get_l1_norm(x_test_float, x_adv_float)
        l2_max, l2_avg = b.get_l2_norm(x_test_float, x_adv_float)
        linf_max, linf_avg = b.get_linf_norm(x_test_float, x_adv_float)
        print("L0 distortion -> Max: " + str(l0_max) + " -> Avg: " + str(l0_avg))
        print("L1 distortion -> Max: " + str(l1_max) + " -> Avg: " + str(l1_avg))
        print("L2 distortion -> Max: " + str(l2_max) + " -> Avg: " + str(l2_avg))
        print("Linf distortion -> Max: " + str(linf_max) + " -> Avg: " + str(linf_avg) + "\n")

    # Plot benign samples, maximum noise, and most adversarial samples
    #b.plot_adv_img(x_test_f32, x_adv_f32, 20)


if __name__ == '__main__':
    main()
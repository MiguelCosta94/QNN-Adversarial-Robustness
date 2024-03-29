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
    parser.add_argument("--ann_float", type=str, default=None, help="path to the floating-point ANN")
    parser.add_argument("--qnn_int16", type=str, default=None, help="path to the int-16 QNN")
    parser.add_argument("--qnn_int8", type=str, default=None, help="path to the int-8 QNN")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'num_classes', 'logs_dir', 'ann_float', 'qnn_int16', 'qnn_int8']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")
        
    run_cfgs = vars(cfgs)

    return run_cfgs


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
    ann_float = tf.keras.models.load_model(cfgs['ann_float'])
    qnn_int16 = b.get_ml_quant_model(cfgs['qnn_int16'])
    scaler_int16, zp_int16 = b.get_input_quant_details(qnn_int16)
    qnn_int8 = b.get_ml_quant_model(cfgs['qnn_int8'])
    scaler_int8, zp_int8 = b.get_input_quant_details(qnn_int8)

    # Step 3: Load test dataset and evaluate accuracy
    test_data_file = cfgs['logs_dir'] + '/x_test_float.bin'
    test_labels_file = cfgs['logs_dir'] + '/labels.bin'
    test_data_file = os.path.join(os.path.dirname(__file__), test_data_file)
    test_labels_file = os.path.join(os.path.dirname(__file__), test_labels_file)
    ## CIFAR-10 -> float64; VWW -> float32; COFFEE -> float32
    x_test_float = np.fromfile(test_data_file, dtype=np.float32)
    y_test = np.fromfile(test_labels_file, dtype=np.int8)
    x_test_float = np.reshape(x_test_float, input_shape)
    y_test = tf.one_hot(y_test, cfgs['num_classes']).numpy()

    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    accuracy = b.get_accuracy_f32(ann_float, x_test_float, y_test)
    print("Float -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_int16, x_test_int16, y_test)
    print("Int16 -> ACC: " + str(accuracy))
    accuracy = b.get_accuracy_quant_model(qnn_int8, x_test_int8, y_test)
    print("Int8 -> ACC: " + str(accuracy))

    # Step 4: Load adversarial dataset and evaluate accuracy
    for x_adv_file in sorted(glob.glob(cfgs['logs_dir'] + '/adv_float*.bin')):
        print("##############")
        print(str(x_adv_file))
        x_adv_float = np.fromfile(x_adv_file, dtype=np.float32)
        x_adv_float = np.reshape(x_adv_float, input_shape)
        x_adv_int16 = b.quantize_dataset_int16(x_adv_float, scaler_int16, zp_int16)
        x_adv_int8 = b.quantize_dataset_int8(x_adv_float, scaler_int8, zp_int8)

        accuracy = b.get_accuracy_f32(ann_float, x_adv_float, y_test)
        print("Float -> ACC: " + str(accuracy))
        accuracy = b.get_accuracy_quant_model(qnn_int16, x_adv_int16, y_test)
        print("Int16 -> ACC: " + str(accuracy))
        accuracy = b.get_accuracy_quant_model(qnn_int8, x_adv_int8, y_test)
        print("Int8 -> ACC: " + str(accuracy))

        l0_max, l0_avg = b.get_l0_norm(x_test_float, x_adv_float)
        l1_max, l1_avg = b.get_l1_norm(x_test_float, x_adv_float)
        l2_max, l2_avg = b.get_l2_norm(x_test_float, x_adv_float)
        linf_max, linf_avg = b.get_linf_norm(x_test_float, x_adv_float)
        print("L0 distortion -> Max: " + str(l0_max) + " -> Avg: " + str(l0_avg))
        print("L1 distortion -> Max: " + str(l1_max) + " -> Avg: " + str(l1_avg))
        print("L2 distortion -> Max: " + str(l2_max) + " -> Avg: " + str(l2_avg))
        print("Linf distortion -> Max: " + str(linf_max) + " -> Avg: " + str(linf_avg) + "\n")

    # Plot benign samples, maximum noise, and most adversarial samples
    #b.plot_adv_img(x_test_float, x_adv_float, 20)


if __name__ == '__main__':
    main()
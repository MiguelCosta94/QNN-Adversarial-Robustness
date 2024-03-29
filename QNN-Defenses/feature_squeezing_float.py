import sys
import logging
import tensorflow as tf
import numpy as np
import glob
from argparse import ArgumentParser
import utils.backend as b
import utils.dataset_loader as ds
import defenses.feature_squeezing_backend as ft


def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10, vww, or coffee")
    parser.add_argument("--ann_source_float", type=str, default=None, help="path to the source floating-point ANN")
    parser.add_argument("--datasets_path", type=str, default=None, help="path to the folder with the test and adversarial datasets")
    parser.add_argument("--ft_calculate_threshold", type=str, default=False, help="calculate the detection threshold or not\
                                                                                                        If not, set --ft_treshold")
    parser.add_argument("--ft_threshold", type=float, default=1.7634952, help="detection threshold. If --ft_calculate_threshold is\
                                                                                                        set this value is discarded")
    parser.add_argument("--ft_fpr", type=float, default=0.05, help="false positive rate - used in threshold calculation")
    parser.add_argument("--ft_color_bit", type=int, default=4, help="color bit depth")
    parser.add_argument("--ft_smooth_window_size", type=int, default=2, help="size of the smoothing window")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'ann_source_float', 'datasets_path']
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
        num_classes = 10
    elif cfgs['dataset_id'] == 'vww':
        input_shape = (-1,96,96,3)
        num_classes = 2
    elif cfgs['dataset_id'] == 'coffee':
        input_shape = (-1,224,224,3)
        num_classes = 4

    # Step 2: Load source ANN
    ann = tf.keras.models.load_model(cfgs['ann_source_float'])

    # Step 3: Load test and adversarial datasets
    x_test_file = cfgs['datasets_path'] + 'x_test_float.bin'
    y_test_file = cfgs['datasets_path'] + 'labels.bin'
    ## CIFAR-10 -> float64; VWW -> float32; COFFEE -> float32
    x_test_float = np.fromfile(x_test_file, dtype=np.float64)
    x_test_float = np.reshape(x_test_float, input_shape)
    y_test = np.fromfile(y_test_file, dtype=np.int8)
    y_test = tf.one_hot(y_test, num_classes).numpy()

    # Step 4: Evaluate the old ANN
    accuracy_n1 = b.get_accuracy_f32(ann, x_test_float, y_test)
    print("TEST -> Accuracy: {}%".format(accuracy_n1 * 100))

    # Step 5: Apply and evaluate Feature Squeezing
    # Get detection threshold
    if cfgs['ft_calculate_threshold'] == "True":
        if cfgs['dataset_id'] == 'cifar10':
            x_train_float, y_train = ds.get_cifar10_train_ds_f32()
        elif cfgs['dataset_id'] == 'vww':
            x_train_float, y_train = ds.get_vww_train_ds_f32()
        elif cfgs['dataset_id'] == 'coffee':
            x_train_float, y_train = ds.get_coffee_train_ds_f32()

        threshold = ft.get_threshold(x_train_float, y_train, cfgs['ft_fpr'], cfgs['ft_color_bit'],
                                                        cfgs['ft_smooth_window_size'], ann, num_classes)
    else:
        threshold = cfgs['ft_threshold']

    for x_adv_file in sorted(glob.glob(cfgs['datasets_path'] + 'adv_float*.bin')):
        print("\n" + str(x_adv_file))
        x_adv = np.fromfile(x_adv_file, dtype=np.float32)
        x_adv = np.reshape(x_adv, input_shape)
        
        # Get squeezed data
        x_clean_1 = ft.squeeze_data(x_adv, y_test, cfgs['ft_color_bit'], cfgs['ft_smooth_window_size'])

        # Detect adversarial examples and evaluate the defense
        accuracy_n2, preds_n2 = b.get_accuracy_preds_f32(ann, x_adv, y_test)
        print("ADV -> Accuracy: {}%".format(accuracy_n2 * 100))
        accuracy_n3, preds_n3 = b.get_accuracy_preds_f32(ann, x_clean_1, y_test)
        print("SQUEEZE -> Accuracy: {}%".format(accuracy_n3 * 100))

        #x_clean_2, y_clean = ft.remove_adv_examples(x_clean_1, y_test, preds_n2, preds_n3, threshold)
        #accuracy_n4 = b.get_accuracy_f32(ann, x_clean_2, y_clean)
        #print("ACC -> Clean: " + str(accuracy_n4))
        #
        #if accuracy_n1 - accuracy_n2 == 0:
        #    detection_rate = 1
        #else:
        #    detection_rate = (accuracy_n3 - accuracy_n2) / (accuracy_n1 - accuracy_n2)
        #    detection_rate = max(detection_rate, 0)
        #print("DR: " + str(detection_rate))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main()
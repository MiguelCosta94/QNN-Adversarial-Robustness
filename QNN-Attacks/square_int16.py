import tensorflow as tf
from argparse import ArgumentParser
import utils.backend as b
import utils.dataset_loader as ds
from tflite_to_cmsis import tflite_to_cmsis_main as cm
from art.attacks.evasion.square_attack_int16 import SquareAttack_Int16
from art.estimators.classification.tensorflow_int16 import TensorFlowV2Classifier_Int16


def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10, vww, or coffee")
    parser.add_argument("--dataset_size", type=int, default=1000, help="number of samples to perturb")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes in target models")
    parser.add_argument("--qnn_int16", type=str, default=None, help="path to the int-16 QNN")
    parser.add_argument("--logs_folder", type=str, default=None, help="path to save the adversarial examples")
    parser.add_argument("--square_norm", type=str, default=2, help="")
    parser.add_argument("--square_max_iter", type=int, default=2000, help="")
    parser.add_argument("--square_eps", type=float, default=1.0, help="")
    parser.add_argument("--square_p_init", type=float, default=0.8, help="")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'num_classes', 'qnn_int16']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")
        
    run_cfgs = vars(cfgs)
    if run_cfgs['square_norm'] != "inf":
        run_cfgs['square_norm'] = int(run_cfgs['square_norm'])

    return run_cfgs


def main():
    # Step 1: Load configs
    cfgs = load_configs()

    if cfgs['dataset_id'] == 'cifar10':
        input_shape = (32,32,3)
    elif cfgs['dataset_id'] == 'vww':
        input_shape = (96,96,3)
    elif cfgs['dataset_id'] == 'coffee':
        input_shape = (224,224,3)

    # Step 2: Load ANN/QNNs
    qnn_int16 = b.get_ml_quant_model(cfgs['qnn_int16'])
    scaler_int16, zp_int16 = b.get_input_quant_details(qnn_int16)
    art_classifier = TensorFlowV2Classifier_Int16(model=qnn_int16, clip_values=(0, 1),
                            nb_classes=cfgs['num_classes'], input_shape=input_shape,
				            loss_object=tf.keras.losses.CategoricalCrossentropy())

    # Step 3: Load dataset and generate .bin files
    if cfgs['dataset_id'] == 'cifar10':
        x_test_float, y_test = ds.get_cifar10_test_ds_f32()
    elif cfgs['dataset_id'] == 'vww':
        x_test_float, y_test = ds.get_vww_test_ds_f32()
    elif cfgs['dataset_id'] == 'coffee':
        x_test_float, y_test = ds.get_coffee_test_ds_f32()

    x_test_float, y_test = x_test_float[0:cfgs['dataset_size']], y_test[0:cfgs['dataset_size']]
    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    cm.generate_dataset_bin(x_test_int16, cfgs['logs_folder'], 'x_test_int16.bin')
    cm.generate_labels_bin(y_test, cfgs['logs_folder'], 'labels.bin')

    # Step 4: Evaluate classifiers on benign test examples
    accuracy = b.get_accuracy_quant_model(qnn_int16, x_test_int16, y_test)
    print("Int16 -> Accuracy on benign test examples: {}%".format(accuracy * 100) + "\n")

    # Step 5: Generate adversarial test examples
    attack = SquareAttack_Int16(estimator=art_classifier, norm=cfgs['square_norm'], max_iter=cfgs['square_max_iter'],
                          eps=cfgs['square_eps'], p_init=cfgs['square_p_init'], batch_size=1)
    x_adv_float = attack.generate(x=x_test_float, y=y_test)
    x_adv_int16 = b.quantize_dataset_int16(x_adv_float, scaler_int16, zp_int16)

    # Step 6: Evaluate the classifiers on adversarial test examples
    accuracy = b.get_accuracy_quant_model(qnn_int16, x_adv_int16, y_test)
    print("Int16 -> Accuracy: {}%".format(accuracy * 100))

    # Step 7: Calculate distance metrics
    l0_max, l0_avg = b.get_l0_norm(x_test_float, x_adv_float)
    l1_max, l1_avg = b.get_l1_norm(x_test_float, x_adv_float)
    l2_max, l2_avg = b.get_l2_norm(x_test_float, x_adv_float)
    linf_max, linf_avg = b.get_linf_norm(x_test_float, x_adv_float)
    print("L0 distortion -> Max: " + str(l0_max) + " -> Avg: " + str(l0_avg))
    print("L1 distortion -> Max: " + str(l1_max) + " -> Avg: " + str(l1_avg))
    print("L2 distortion -> Max: " + str(l2_max) + " -> Avg: " + str(l2_avg))
    print("Linf distortion -> Max: " + str(linf_max) + " -> Avg: " + str(linf_avg) + "\n")

    ## Step 8: Save adversarial dataset to .bin file
    cm.generate_dataset_bin(x_adv_int16, cfgs['logs_folder'], 'adv_int16.bin')

    # Step 9: Plot benign samples, maximum noise, and most adversarial samples
    #b.plot_adv_img(x_test_float, x_adv_float, 20)


if __name__ == '__main__':
    main()
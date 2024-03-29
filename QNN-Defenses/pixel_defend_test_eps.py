import tensorflow as tf
import numpy as np
import glob
from argparse import ArgumentParser
import utils.backend as b
from art.defences.preprocessor import PixelDefend
from pixel_cnn.model import bits_per_dim_loss


def load_configs():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset_id", type=str, default=None, help="dataset to use: cifar10, vww, or coffee")
    parser.add_argument("--ann_source_float", type=str, default=None, help="path to the source floating-point ANN")
    parser.add_argument("--datasets_path", type=str, default=None, help="path to the folder with the test and adversarial datasets")
    parser.add_argument("--pixel_cnn", type=str, default=None, help="path to the Pixel CNN")
    parser.add_argument("--pd_eps_list", type=str, default="38", help="")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'ann_source_float', 'datasets_path', 'pixel_cnn', 'pd_eps_list']
    for arg_name in required_cfgs:
        if getattr(cfgs, arg_name) is None:
            raise ValueError(f"Required argument {arg_name} is not set.")

    run_cfgs = vars(cfgs)
    pd_eps_list = run_cfgs['pd_eps_list'].split(",")
    run_cfgs['pd_eps_list'] = np.array(pd_eps_list, dtype=int)

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

    # Step 3: Load Pixel CNN
    pixel_cnn = tf.keras.models.load_model(cfgs['pixel_cnn'], custom_objects={'bits_per_dim_loss': bits_per_dim_loss})
    pixel_cnn.summary()

    # Step 4: Load labels
    y_test_file = cfgs['datasets_path'] + 'labels.bin'
    y_test = np.fromfile(y_test_file, dtype=np.int8)
    y_test = tf.one_hot(y_test, num_classes).numpy()

    #######################################################################
    # Step 4: Test EPS against adversarial examples
    for eps in cfgs['pd_eps_list']:
        # Setup Pixel Defend
        print("\nEPS {}".format(eps))
        eps = int(eps)
        pixel_defend = PixelDefend(clip_values=(0,1), eps=eps, pixel_cnn=pixel_cnn, batch_size=1, apply_predict=False, verbose=True)

        for x_adv_file in sorted(glob.glob(cfgs['datasets_path'] + 'adv_float*.bin')):
            print("\n" + str(x_adv_file))
            x_adv = np.fromfile(x_adv_file, dtype=np.float32)
            x_adv = np.reshape(x_adv, input_shape)
            
            # Get clean data
            x_clean, y_test = pixel_defend(x_adv, y_test)

            # Evaluate the defense
            accuracy = b.get_accuracy_f32(ann, x_adv, y_test)
            print("ADV -> Accuracy: {}%".format(accuracy * 100))
            accuracy = b.get_accuracy_f32(ann, x_clean, y_test)
            print("CLEAN -> Accuracy: {}%".format(accuracy * 100))


if __name__ == '__main__':
#   python3 pixel_defend_test_eps.py --dataset_id cifar10 --ann_source_float models/cifar_resnet_float.h5 \
#   --datasets_path "../QNN-Attacks/logs/cifar/fgsm_pixel_defend/" --pixel_cnn pixel_cnn/px_cnn/pixel_cnn_cifar --pd_eps_list "38,48"

    main()

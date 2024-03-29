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
    parser.add_argument("--pd_eps", type=int, default=38, help="")

    cfgs = parser.parse_args()

    # Check if any required argument is not set
    required_cfgs = ['dataset_id', 'ann_source_float', 'datasets_path', 'pixel_cnn']
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

    # Step 5: Apply and evaluate Pixel Defend
    # Load Pixel CNN
    pixel_cnn = tf.keras.models.load_model(cfgs['pixel_cnn'], custom_objects={'bits_per_dim_loss': bits_per_dim_loss})
    #pixel_cnn.summary()

    # Setup PixelDefend
    pixel_defend = PixelDefend(clip_values=(0,1), eps=cfgs['pd_eps'], pixel_cnn=pixel_cnn, batch_size=1, apply_predict=False, verbose=True)

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
    main()
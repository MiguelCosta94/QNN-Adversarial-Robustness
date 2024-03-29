import tensorflow as tf
import numpy as np


def convert_to_tflite_int8(ann_float, dataset_id, x_test_float):
    if dataset_id == 'cifar10':
        def representative_dataset_generator():
            idx = np.load('tiny/cifar_calibration_samples_idxs.npy')
            for i in idx:
                sample_img = np.expand_dims(np.array(x_test_float[i], dtype=np.float32), axis=0)
                yield [sample_img]
    
    elif dataset_id == 'vww':
        def representative_dataset_generator():
            for idx, image in enumerate(x_test_float):
                # 10 representative images should be enough for calibration.
                if idx > 10:
                    return
                yield [image.reshape(1, 96, 96, 3)]

    elif dataset_id == 'coffee':
        def representative_dataset_generator():
            for img in x_test_float:
                img = np.reshape(img, (1, 224,224,3))
                yield [img]

    converter = tf.lite.TFLiteConverter.from_keras_model(ann_float)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    return tflite_model


def convert_to_tflite_int16(ann_float, dataset_id, x_test_float):
    if dataset_id == 'cifar10':
        def representative_dataset_generator():
            idx = np.load('tiny/cifar_calibration_samples_idxs.npy')
            for i in idx:
                sample_img = np.expand_dims(np.array(x_test_float[i], dtype=np.float32), axis=0)
                yield [sample_img]

    elif dataset_id == 'vww':
        def representative_dataset_generator():
            for idx, image in enumerate(x_test_float):
                # 10 representative images should be enough for calibration.
                if idx > 10:
                    return
                yield [image.reshape(1, 96, 96, 3)]

    elif dataset_id == 'coffee':
        def representative_dataset_generator():
            for img in x_test_float:
                img = np.reshape(img, (1, 224,224,3))
                yield [img]

    converter = tf.lite.TFLiteConverter.from_keras_model(ann_float)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    tflite_model = converter.convert()

    return tflite_model
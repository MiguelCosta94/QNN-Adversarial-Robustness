import tensorflow as tf
import utils.backend as b
import utils.dataset_loader as ds
import numpy as np


def calculate_avg_zero_density(grad_matrix):
    grad_matrix = grad_matrix.reshape(grad_matrix.shape[0], -1)
    
    zero_densitty = []
    for grad_vector in grad_matrix:
        total_elements = len(grad_vector)
        num_zeroes = np.count_nonzero(grad_vector == 0)
        percentage = (num_zeroes / total_elements) * 100
        zero_densitty.append(percentage)

    zero_densitty = np.array(zero_densitty)
    avg_zero_density = np.mean(zero_densitty)

    return avg_zero_density
   
# Function to calculate categorical cross-entropy loss
def categorical_cross_entropy_loss(targets, predictions):
    predictions = tf.cast(predictions, tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(targets, predictions, from_logits=False)
    return loss

# Function to estimate gradients using finite differences
def estimate_gradient(model, inputs, targets, epsilon=1e-5):
    gradients = np.zeros_like(inputs, dtype=np.float32)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(inputs.shape[2]):
                for l in range(inputs.shape[3]):
                    inputs_plus_epsilon = inputs.copy()
                    inputs_plus_epsilon[i, j, k, l] += epsilon
                    loss_plus = np.mean(categorical_cross_entropy_loss(targets, model(inputs_plus_epsilon)))

                    inputs_minus_epsilon = inputs.copy()
                    inputs_minus_epsilon[i, j, k, l] -= epsilon
                    loss_minus = np.mean(categorical_cross_entropy_loss(targets, model(inputs_minus_epsilon)))

                    gradients[i, j, k, l] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradients

def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    def tflite_model(inputs):
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        outputs = interpreter.get_tensor(output_index)
        return outputs

    return tflite_model


def main(cfgs):
    ###LOAD MODELS###
    ann_float = tf.keras.models.load_model(cfgs['ann_float'])
    qnn_int16 = load_tflite_model(cfgs['qnn_int16'])
    qnn_int8 = load_tflite_model(cfgs['qnn_int8'])

    ###LOAD SCALERS###
    itp_int16 = b.get_ml_quant_model(cfgs['qnn_int16'])
    itp_int8 = b.get_ml_quant_model(cfgs['qnn_int8'])
    scaler_int16, zp_int16 = b.get_input_quant_details(itp_int16)
    scaler_int8, zp_int8 = b.get_input_quant_details(itp_int8)

    ###LOAD DATASET###
    if cfgs['dataset_id'] == 'cifar10':
        x_test_float, y_test = ds.get_cifar10_test_ds_f32()
    elif cfgs['dataset_id'] == 'vww':
        x_test_float, y_test = ds.get_vww_test_ds_f32()
    elif cfgs['dataset_id'] == 'coffee':
        x_test_float, y_test = ds.get_coffee_test_ds_f32()

    x_test_float, y_test = x_test_float[0:cfgs['dataset_size']], y_test[0:cfgs['dataset_size']]
    x_test_int16 = b.quantize_dataset_int16(x_test_float, scaler_int16, zp_int16)
    x_test_int8 = b.quantize_dataset_int8(x_test_float, scaler_int8, zp_int8)

    ###CALCULATE AVG ZERO DENSITY OF FLOAT-32 MODEL###
    gradient_float = estimate_gradient(ann_float, x_test_float, y_test)
    avg_zero_density = calculate_avg_zero_density(gradient_float)
    print("FLOAT -> AVG ZERO DENSITY: ", avg_zero_density)

    ###CALCULATE AVG ZERO DENSITY OF INT-16 MODEL###
    gradients = []
    for sample, label in zip(x_test_int16, y_test):
        sample = np.expand_dims(sample, axis=0)
        label = np.expand_dims(label, axis=0)
        gradient_int16 = estimate_gradient(qnn_int16, sample, label)
        gradients.append(gradient_int16)
    gradients = np.array(gradients)
    avg_zero_density = calculate_avg_zero_density(gradients)
    print("INT-16 -> AVG ZERO DENSITY: ", avg_zero_density)

    ###CALCULATE AVG ZERO DENSITY OF INT-8 MODEL###
    gradients = []
    for sample, label in zip(x_test_int8, y_test):
        sample = np.expand_dims(sample, axis=0)
        label = np.expand_dims(label, axis=0)
        gradient_int8 = estimate_gradient(qnn_int8, sample, label)
        gradients.append(gradient_int8)
    gradients = np.array(gradients)
    avg_zero_density = calculate_avg_zero_density(gradients)
    print("INT-8 -> AVG ZERO DENSITY: ", avg_zero_density)


if __name__ == '__main__':
    cfgs = {'dataset_id': 'cifar10', 'dataset_size': 1000, 'ann_float': 'models/cifar_resnet_float.h5',
            'qnn_int16': 'models/cifar_resnet_int16.tflite', 'qnn_int8': 'models/cifar_resnet_int8.tflite'}

    #cfgs = {'dataset_id': 'vww', 'dataset_size': 1000, 'ann_float': 'models/vww_float.h5',
    #        'qnn_int16': 'models/vww_int16.tflite', 'qnn_int8': 'models/vww_int8.tflite'}
#
    #cfgs = {'dataset_id': 'coffee', 'dataset_size': 1000, 'ann_float': 'models/coffee_cnn_float.h5',
    #        'qnn_int16': 'models/coffee_cnn_int16.tflite', 'qnn_int8': 'models/coffee_cnn_int8.tflite'}

    main(cfgs)
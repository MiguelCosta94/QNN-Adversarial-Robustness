import numpy as np
import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from tflite_to_cmsis import tfl_reader as tflr


def get_ml_quant_model(ml_model_dir):
    interpreter = tf.lite.Interpreter(model_path=ml_model_dir)
    interpreter.allocate_tensors()  # Needed before execution!

    return interpreter


def get_ml_model_int8_dict(ml_model_dir):
    with open(ml_model_dir, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
    model_dict = tflr.CreateDictFromFlatbuffer(file_data)

    return model_dict


def get_input_quant_details(interpreter):
    input_details = interpreter.get_input_details()
    in_scale = input_details[0]['quantization'][0]
    in_zp = input_details[0]['quantization'][1]

    return in_scale, in_zp


def quantize_dataset_int8(dataset, scaler, zp):
    dataset_int8 = dataset / scaler + zp
    dataset_int8 = round_half_up(dataset_int8, 0)
    dataset_int8 = np.array(dataset_int8, dtype=np.int8)
    dataset_int8 = np.clip(dataset_int8, a_min=-128, a_max=127)

    return dataset_int8


def quantize_dataset_int16(dataset, scaler, zp):
    dataset_int16 = dataset / scaler + zp
    dataset_int16 = round_half_up(dataset_int16, 0)
    dataset_int16 = np.array(dataset_int16, dtype=np.int16)
    dataset_int16 = np.clip(dataset_int16, a_min=-32768, a_max=32767)

    return dataset_int16


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n*multiplier + 0.5) / multiplier


def clone_ann_without_softmax(ann):
    ann_no_softmax = tf.keras.models.clone_model(ann)
    ann_no_softmax.layers[-1].activation = tf.keras.activations.linear
    ann_no_softmax.compile(optimizer='Adam', loss='categorical_crossentropy')
    ann_no_softmax.set_weights(ann.get_weights())

    return ann_no_softmax


def get_accuracy_f32(ann, x_test, y_test):
    predictions = ann.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

    return accuracy


def get_accuracy_preds_f32(ann, x_test, y_test):
    predictions = ann.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

    return accuracy, predictions


def get_accuracy_quant_model(ml_model_int8, x_test, y_test):
    input_details = ml_model_int8.get_input_details()
    output_details = ml_model_int8.get_output_details()
    predictions = []

    for idx in range(x_test.shape[0]):
        input_data = x_test[idx].reshape(input_details[0]['shape'])
        ml_model_int8.set_tensor(input_details[0]['index'], input_data)
        ml_model_int8.invoke()
        output_data = ml_model_int8.get_tensor(output_details[0]['index'])
        predictions.append(output_data.reshape(output_details[0]['shape'][1], ))

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

    return accuracy


def get_accuracy_preds_quant_model(ml_model_int8, x_test, y_test):
    input_details = ml_model_int8.get_input_details()
    output_details = ml_model_int8.get_output_details()
    predictions = []

    for idx in range(x_test.shape[0]):
        input_data = x_test[idx].reshape(input_details[0]['shape'])
        ml_model_int8.set_tensor(input_details[0]['index'], input_data)
        ml_model_int8.invoke()
        output_data = ml_model_int8.get_tensor(output_details[0]['index'])
        predictions.append(output_data.reshape(output_details[0]['shape'][1], ))

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

    return accuracy, predictions


def get_l0_norm(benign_data, adv_data):
    benign_data = np.reshape(benign_data, newshape=(benign_data.shape[0], -1))
    adv_data = np.reshape(adv_data, newshape=(adv_data.shape[0], -1))

    l0_norm = np.linalg.norm(benign_data-adv_data, ord=0, axis=-1)
    l0_max = np.max(l0_norm)
    l0_avg = np.mean(l0_norm)

    return l0_max, l0_avg


def get_l1_norm(benign_data, adv_data):
    benign_data = np.reshape(benign_data, newshape=(benign_data.shape[0], -1))
    adv_data = np.reshape(adv_data, newshape=(adv_data.shape[0], -1))

    l1_norm = np.linalg.norm(benign_data-adv_data, ord=1, axis=-1)
    l1_max = np.max(l1_norm)
    l1_avg = np.mean(l1_norm)

    return l1_max, l1_avg


def get_l2_norm(benign_data, adv_data):
    benign_data = np.reshape(benign_data, newshape=(benign_data.shape[0], -1))
    adv_data = np.reshape(adv_data, newshape=(adv_data.shape[0], -1))

    l2_norm = np.linalg.norm(benign_data-adv_data, ord=2, axis=-1)
    l2_max = np.max(l2_norm)
    l2_avg = np.mean(l2_norm)

    return l2_max, l2_avg


def get_linf_norm(benign_data, adv_data):
    benign_data = np.reshape(benign_data, newshape=(benign_data.shape[0], -1))
    adv_data = np.reshape(adv_data, newshape=(adv_data.shape[0], -1))

    linf_norm = np.linalg.norm(benign_data-adv_data, ord=np.inf, axis=-1)
    linf_max = np.max(linf_norm)
    linf_avg = np.mean(linf_norm)

    return linf_max, linf_avg


def plot_adv_img(ben_data, adv_data, num_samples):
    per_data = ben_data - adv_data
    per_data = scale_to_unit_norm(per_data)

    matplotlib.use('TkAgg')
    num_images = num_samples * 3
    rows = (num_images - 1) // 10 + 1

    fig, axes = plt.subplots(rows, 10, figsize=(20, rows*2))

    for i in range(num_images):
        if i < num_samples:
            data = np.array(ben_data[i])
        elif i >= num_samples and i < num_samples*2:
            data = np.array(per_data[i - num_samples])
        else:
            data = np.array(adv_data[i - num_samples*2])

        row = i // 10
        col = i % 10
        axes[row, col].imshow(data)

    plt.show()


def scale_to_unit_norm(data):
    shape = data.shape
    data = data.reshape(data.shape[0], -1)
    transformer = MaxAbsScaler().fit(data)
    data = transformer.transform(data)
    data = data.reshape(shape)

    return data
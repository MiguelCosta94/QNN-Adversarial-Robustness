import numpy as np
from tflite_to_cmsis import tfl_reader as tflr
from tflite_to_cmsis import tfl_param_extract as pe
from tflite_to_cmsis import c_code_gen as cg
from tflite_to_cmsis import buffer_mapping as bm


def generate_dataset_bin(data, dir, data_filename):
    for i, row in enumerate(data):
        row_bin = row.tobytes()

        if i == 0:
            data_byte = bytearray(row_bin)
        else:
            data_byte.extend(row_bin)

    cg.write_to_bin_file(data_byte, dir, data_filename)


def generate_labels_bin(labels, dir, labels_filename):
    labels_int = np.argmax(labels, axis=-1)
    labels_int = np.array(labels_int, dtype=np.int8)
    labels_byte = bytearray(labels_int)
    cg.write_to_bin_file(labels_byte, dir, labels_filename)


def generate_c_code(model_dict, arm_specs, dataset_details, mode, dir):
    model_id = dataset_details['id']

    tensors = model_dict['subgraphs'][0]['tensors']
    tensors = tflr.tensors_name_to_ascii(tensors, model_id)
    operators = model_dict['subgraphs'][0]['operators']
    buffers = model_dict['buffers']
    ann_input_tensors, filter_tensor, bias_tensor, output_tensor = tflr.get_operator_tensors(operators[0], tensors)
    ann_input_tensor = ann_input_tensors[0]
    input_tensors, filter_tensor, bias_tensor, ann_output_tensor = tflr.get_operator_tensors(operators[-1], tensors)

    if model_id == 'cif':
        cmsis_nn_buffers_map = bm.get_buf_io_map_resnet(operators, tensors)
    else:
        cmsis_nn_buffers_map = bm.get_buf_io_map_std(operators, tensors)

    cmsis_nn_params = pe.get_cmsis_nn_params(tensors, operators, buffers)
    cg.generate_nn_params_h(cmsis_nn_params, ann_input_tensor['quantization'],
                        ann_output_tensor['quantization'], arm_specs, dataset_details, dir)
    cg.generate_nn_wt_bias_h(cmsis_nn_params, dir)
    cg.generate_main_c(cmsis_nn_params, cmsis_nn_buffers_map, model_id, mode, dir)
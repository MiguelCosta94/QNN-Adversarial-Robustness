import os
import sys
import numpy as np
import pathlib
from tflite_to_cmsis import gen_params_header as gp
from tflite_to_cmsis import gen_main_c as gm


def generate_nn_params_h(cmsis_nn_params, input_quant_details, output_quant_details, arm_specs, dataset_details, dir):
    conv_cnt = 0
    dw_conv_cnt = 0
    avg_pool_cnt = 0
    fc_cnt = 0
    softmax_cnt = 0
    add_cnt = 0

    file_string = gp.get_c_code_param_defines(cmsis_nn_params, input_quant_details,
                                            output_quant_details, arm_specs, dataset_details)

    for param_set in cmsis_nn_params:
        if 'conv2d' in param_set['layer']:
            code_string = gp.get_c_code_param_conv(param_set, '_conv' + str(conv_cnt))
            file_string += code_string
            conv_cnt = conv_cnt + 1
        elif 'batch_normalization' in param_set['layer']:
            code_string = gp.get_c_code_param_dw_conv(param_set, '_dw_cnv' + str(dw_conv_cnt))
            file_string += code_string
            dw_conv_cnt = dw_conv_cnt + 1
        elif 'average_pooling2d' in param_set['layer']:
            code_string = gp.get_c_code_param_avg_pool(param_set, '_avg_pool' + str(avg_pool_cnt))
            file_string += code_string
            avg_pool_cnt = avg_pool_cnt + 1
        elif 'dense' in param_set['layer']:
            code_string = gp.get_c_code_param_fc(param_set, '_fc' + str(fc_cnt))
            file_string += code_string
            fc_cnt = fc_cnt + 1
        elif 'Identity' in param_set['layer']:
            code_string = gp.get_c_code_param_softmax(param_set, '_softmax' + str(softmax_cnt))
            file_string += code_string
            softmax_cnt = softmax_cnt + 1
        elif 'add' in param_set['layer']:
            code_string = gp.get_c_code_param_add(param_set, '_add' + str(add_cnt))
            file_string += code_string
            add_cnt += 1

    file_string += (
        '#endif'
    )

    write_to_file(file_string, dir, 'nn_params.h')


def generate_nn_wt_bias_h(cmsis_nn_params, dir):
    conv_cnt = 0
    dw_conv_cnt = 0
    fc_cnt = 0

    file_string = gp.get_c_code_wt_bias_defines()

    for param_set in cmsis_nn_params:
        if 'conv2d' in param_set['layer']:
            code_string = gp.get_c_code_wt_bias(param_set, '_conv' + str(conv_cnt))
            file_string += code_string
            conv_cnt = conv_cnt + 1
        elif 'batch_normalization' in param_set['layer']:
            code_string = gp.get_c_code_wt_bias(param_set, '_dw_cnv' + str(dw_conv_cnt))
            file_string += code_string
            dw_conv_cnt = dw_conv_cnt + 1
        elif 'dense' in param_set['layer']:
            code_string = gp.get_c_code_wt_bias(param_set, '_fc' + str(fc_cnt))
            file_string += code_string
            fc_cnt = fc_cnt + 1

    file_string += (
        '#endif'
    )

    write_to_file(file_string, dir, 'nn_wt.h')


def generate_main_c(cmsis_nn_params, cmsis_nn_buffers_map, model_id, mode, dir):
    buf_size = get_buffers_size(cmsis_nn_params)

    if mode == 'ser':
        file_string = gm.get_c_code_main_function_ser(buf_size, model_id)
    else:
        file_string = gm.get_c_code_main_function_usb(buf_size, model_id)

    file_string += gm.get_c_code_check_ctx_size(cmsis_nn_params)
    
    if model_id == 'ano':
        file_string += gm.get_c_code_dequantize()
        
    file_string += gm.get_c_code_run_ann(cmsis_nn_params, cmsis_nn_buffers_map, model_id)

    write_to_file(file_string, dir, 'main.cpp')

##############################################################################################
############################### AUXILIAR FUNCTIONS ###########################################
def get_buffers_size(cmsis_nn_params):
    classic_layers = ['conv2d', 'batch_normalization', 'dense', 'average_pooling2d']
    buf_size = 0

    for param_set in cmsis_nn_params:
        if any(layer_type in param_set['layer'] for layer_type in classic_layers):
            curr_in_size = param_set['input_dims']['n'] * param_set['input_dims']['h'] * param_set['input_dims']['w'] * param_set['input_dims']['c']
            curr_out_size = param_set['output_dims']['n'] * param_set['output_dims']['h'] * param_set['output_dims']['w'] * param_set['output_dims']['c']
            lst = [buf_size, curr_in_size, curr_out_size]
            buf_size = max(lst)
        elif 'Identity' in param_set['layer']:
            curr_size = param_set['num_rows'] * param_set['row_size']
            lst = [buf_size, curr_size]
            buf_size = max(lst)
        elif 'add' in param_set['layer']:
            curr_size = param_set['block_size']
            lst = [buf_size, curr_size]
            buf_size = max(lst)

    return buf_size


def write_to_file(string, dir, file_name):
    np.set_printoptions(threshold=sys.maxsize)

    dir = str(pathlib.Path(__file__).parent.absolute()) + dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += '/' + file_name

    with open(dir, 'w') as f:
        f.write(string)


def write_to_bin_file(data, dir, file_name):
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/../" +dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += '/' + file_name

    with open(dir, 'wb') as f:
        f.write(data)
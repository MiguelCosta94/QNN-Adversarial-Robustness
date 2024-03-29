import re
import numpy as np


def get_c_code_param_defines(cmsis_nn_params, input_quant_details, output_quant_details, arm_specs, dataset_details):
    input_scale = input_quant_details['scale'][0]
    input_zp = input_quant_details['zero_point'][0]
    output_scale = output_quant_details['scale'][0]
    output_zp = output_quant_details['zero_point'][0]
    input_data_size = cmsis_nn_params[0]['input_dims']['h'] * cmsis_nn_params[0]['input_dims']['w'] * cmsis_nn_params[0]['input_dims']['c']
    num_layers_with_ctx = get_num_layers_with_ctx(cmsis_nn_params)
    ctx_size = get_ctx_size(cmsis_nn_params, arm_specs)

    string = (
        '#ifndef NN_PARAMS_H \n'
        '#define NN_PARAMS_H \n\n'

        '#include "arm_nn_types.h"\n\n'

        '#define DATASET_SIZE ' + str(dataset_details['size']) + '\n'
        '#define BATCH_SIZE ' + str(dataset_details['batch']) + '\n'
        '#define INPUT_DATA_SIZE ' + str(input_data_size) + '\n'
        '#define OUTPUT_DATA_SIZE ' + str(dataset_details['output_size']) + '\n'
        '#define NUM_LAYERS_WITH_CTX ' + str(num_layers_with_ctx) + '\n'
        '#define CTX_SIZE ' + str(ctx_size) + '\n'
        '#define ANN_IN_QUANT_SCALE ' + str(input_scale) + '\n'
        '#define ANN_IN_QUANT_ZP ' + str(input_zp) + '\n'
        '#define ANN_OUT_QUANT_SCALE ' + str(output_scale) + '\n'
        '#define ANN_OUT_QUANT_ZP ' + str(output_zp) + '\n\n'
    )

    return string


def get_c_code_param_conv(param_set, sufix):
    multiplier = np.array2string(np.array(param_set['quant_params']['multiplier'], dtype=np.int32), separator=',')
    multiplier = re.sub('[\[\]\n]', '', multiplier)
    shift = np.array2string(np.array(param_set['quant_params']['shift'], dtype=np.int32), separator=',')
    shift = re.sub('[\[\]\n]', '', shift)

    stride = ('{.w = ' + str(param_set['conv_params']['stride']['w']) + ', .h = ' + str(param_set['conv_params']['stride']['h']) + '}')
    padding = ('{.w = ' + str(param_set['conv_params']['padding']['w']) + ', .h = ' + str(param_set['conv_params']['padding']['h']) + '}')
    dilation = ('{.w = ' + str(param_set['conv_params']['dilation']['w']) + ', .h = ' + str(param_set['conv_params']['dilation']['h']) + '}')
    activation = ('{.min = ' + str(param_set['conv_params']['activation']['min']) + ', .max = ' + str(param_set['conv_params']['activation']['max']) + '}')

    string = (
        'int32_t multiplier' + sufix + '[] = {' + multiplier + '};\n'
        'int32_t shift' + sufix + '[] = {' + shift + '};\n\n'

        
        'cmsis_nn_conv_params conv_params' + sufix + ' = {.input_offset = ' + str(param_set['conv_params']['input_offset']) + ', ' +
                                                        '.output_offset = ' + str(param_set['conv_params']['output_offset']) + ', ' +
                                                        '.stride = ' + stride + ', .padding = ' + padding + ', .dilation = ' + dilation +
                                                        ', .activation = ' + activation + '};\n' +

        'cmsis_nn_per_channel_quant_params quant_params' + sufix + ' = {.multiplier = multiplier' + sufix + ', .shift=shift' + sufix +'};\n'
        'cmsis_nn_dims input_dims' + sufix + ' = {.n = ' + str(param_set['input_dims']['n']) + ', .h = ' + str(param_set['input_dims']['h']) +
                                                    ', .w = ' + str(param_set['input_dims']['w']) + ', .c = ' + str(param_set['input_dims']['c']) + '};\n' +
        'cmsis_nn_dims filter_dims' + sufix + ' = {.n = ' + str(param_set['filter_dims']['n']) + ', .h = ' + str(param_set['filter_dims']['h']) +
                                                    ', .w = ' + str(param_set['filter_dims']['w']) + ', .c = ' + str(param_set['filter_dims']['c']) + '};\n' +
        'cmsis_nn_dims bias_dims' + sufix + ' = {.n = ' + str(param_set['bias_dims']['n']) + ', .h = ' + str(param_set['bias_dims']['h']) +
                                                    ', .w = ' + str(param_set['bias_dims']['w']) + ', .c = ' + str(param_set['bias_dims']['c']) + '};\n' +
        'cmsis_nn_dims output_dims' + sufix + ' = {.n = ' + str(param_set['output_dims']['n']) + ', .h = ' + str(param_set['output_dims']['h']) +
                                                    ', .w = ' + str(param_set['output_dims']['w']) + ', .c = ' + str(param_set['output_dims']['c']) + '};\n'
        'uint32_t output_size' + sufix + ' = ' + str(param_set['output_size']) + ';\n\n\n'
    )

    return string


def get_c_code_param_dw_conv(param_set, sufix):
    multiplier = np.array2string(np.array(param_set['quant_params']['multiplier'], dtype=np.int32), separator=',')
    multiplier = re.sub('[\[\]\n]', '', multiplier)
    shift = np.array2string(np.array(param_set['quant_params']['shift'], dtype=np.int32), separator=',')
    shift = re.sub('[\[\]\n]', '', shift)

    stride = ('{.w = ' + str(param_set['dw_conv_params']['stride']['w']) + ', .h = ' + str(param_set['dw_conv_params']['stride']['h']) + '}')
    padding = ('{.w = ' + str(param_set['dw_conv_params']['padding']['w']) + ', .h = ' + str(param_set['dw_conv_params']['padding']['h']) + '}')
    dilation = ('{.w = ' + str(param_set['dw_conv_params']['dilation']['w']) + ', .h = ' + str(param_set['dw_conv_params']['dilation']['h']) + '}')
    activation = ('{.min = ' + str(param_set['dw_conv_params']['activation']['min']) + ', .max = ' + str(param_set['dw_conv_params']['activation']['max']) + '}')

    string = (
        'int32_t multiplier' + sufix + '[] = {' + multiplier + '};\n' +
        'int32_t shift' + sufix + '[] = {' + shift + '};\n\n' +

        'cmsis_nn_dw_conv_params dw_conv_params' + sufix + ' = {.input_offset = ' + str(param_set['dw_conv_params']['input_offset']) + ', ' +
                                                                '.output_offset = ' + str(param_set['dw_conv_params']['output_offset']) + ', ' +
                                                                '.ch_mult = ' + str(param_set['dw_conv_params']['ch_mult']) + ', ' +
                                                                '.stride = ' + stride + ', .padding = ' + padding + ', ' +
                                                                '.dilation = ' + dilation + ', .activation = ' + activation + '};\n' +

        'cmsis_nn_per_channel_quant_params quant_params' + sufix + ' = {.multiplier = multiplier' + sufix + ', .shift=shift' + sufix +'};\n'
        'cmsis_nn_dims input_dims' + sufix + ' = {.n = ' + str(param_set['input_dims']['n']) + ', .h = ' + str(param_set['input_dims']['h']) +
                                                    ', .w = ' + str(param_set['input_dims']['w']) + ', .c = ' + str(param_set['input_dims']['c']) + '};\n' +
        'cmsis_nn_dims filter_dims' + sufix + ' = {.n = ' + str(param_set['filter_dims']['n']) + ', .h = ' + str(param_set['filter_dims']['h']) +
                                                    ', .w = ' + str(param_set['filter_dims']['w']) + ', .c = ' + str(param_set['filter_dims']['c']) + '};\n' +
        'cmsis_nn_dims bias_dims' + sufix + ' = {.n = ' + str(param_set['bias_dims']['n']) + ', .h = ' + str(param_set['bias_dims']['h']) +
                                                    ', .w = ' + str(param_set['bias_dims']['w']) + ', .c = ' + str(param_set['bias_dims']['c']) + '};\n' +
        'cmsis_nn_dims output_dims' + sufix + ' = {.n = ' + str(param_set['output_dims']['n']) + ', .h = ' + str(param_set['output_dims']['h']) +
                                                    ', .w = ' + str(param_set['output_dims']['w']) + ', .c = ' + str(param_set['output_dims']['c']) + '};\n\n\n'
    )

    return string


def get_c_code_param_avg_pool(param_set, sufix):
    stride = ('{.w = ' + str(param_set['pool_params']['stride']['w']) + ', .h = ' + str(param_set['pool_params']['stride']['h']) + '}')
    padding = ('{.w = ' + str(param_set['pool_params']['padding']['w']) + ', .h = ' + str(param_set['pool_params']['padding']['h']) + '}')
    activation = ('{.min = ' + str(param_set['pool_params']['activation']['min']) + ', .max = ' + str(param_set['pool_params']['activation']['max']) + '}')

    string = (
        'cmsis_nn_pool_params pool_params' + sufix + ' = {.stride = ' + stride + ', .padding = ' + padding + ', .activation = ' + activation + '};\n' + 
        'cmsis_nn_dims input_dims' + sufix + ' = {.n = ' + str(param_set['input_dims']['n']) + ', .h = ' + str(param_set['input_dims']['h']) +
                                                    ', .w = ' + str(param_set['input_dims']['w']) + ', .c = ' + str(param_set['input_dims']['c']) + '};\n' +
        'cmsis_nn_dims filter_dims' + sufix + ' = {.n = ' + str(param_set['filter_dims']['n']) + ', .h = ' + str(param_set['filter_dims']['h']) +
                                                    ', .w = ' + str(param_set['filter_dims']['w']) + ', .c = ' + str(param_set['filter_dims']['c']) + '};\n' +
        'cmsis_nn_dims output_dims' + sufix + ' = {.n = ' + str(param_set['output_dims']['n']) + ', .h = ' + str(param_set['output_dims']['h']) +
                                                    ', .w = ' + str(param_set['output_dims']['w']) + ', .c = ' + str(param_set['output_dims']['c']) + '};\n\n\n'
    )

    return string


def get_c_code_param_fc(param_set, sufix):
    activation = ('{.min = ' + str(param_set['fc_params']['activation']['min']) + ', .max = ' + str(param_set['fc_params']['activation']['max']) + '}')

    string = (
        'cmsis_nn_fc_params fc_params' + sufix + ' = {.input_offset = ' + str(param_set['fc_params']['input_offset']) + ', ' +
                                                        '.filter_offset = ' + str(param_set['fc_params']['filter_offset']) + ', ' +
                                                        '.output_offset = ' + str(param_set['fc_params']['output_offset']) + ', ' +
                                                        '.activation = ' + activation + '};\n' + 
        
        'cmsis_nn_per_tensor_quant_params quant_params' + sufix + ' = {.multiplier = ' + str(param_set['quant_params']['multiplier'][0]) + ', .shift = ' + str(param_set['quant_params']['shift'][0]) + '};\n' +
        'cmsis_nn_dims input_dims' + sufix + ' = {.n = ' + str(param_set['input_dims']['n']) + ', .h = ' + str(param_set['input_dims']['h']) +
                                                    ', .w = ' + str(param_set['input_dims']['w']) + ', .c = ' + str(param_set['input_dims']['c']) + '};\n' +
        'cmsis_nn_dims filter_dims' + sufix + ' = {.n = ' + str(param_set['filter_dims']['n']) + ', .h = ' + str(param_set['filter_dims']['h']) +
                                                    ', .w = ' + str(param_set['filter_dims']['w']) + ', .c = ' + str(param_set['filter_dims']['c']) + '};\n' +
        'cmsis_nn_dims bias_dims' + sufix + ' = {.n = ' + str(param_set['bias_dims']['n']) + ', .h = ' + str(param_set['bias_dims']['h']) +
                                                    ', .w = ' + str(param_set['bias_dims']['w']) + ', .c = ' + str(param_set['bias_dims']['c']) + '};\n' +
        'cmsis_nn_dims output_dims' + sufix + ' = {.n = ' + str(param_set['output_dims']['n']) + ', .h = ' + str(param_set['output_dims']['h']) +
                                                    ', .w = ' + str(param_set['output_dims']['w']) + ', .c = ' + str(param_set['output_dims']['c']) + '};\n\n\n'
    )

    return string


def get_c_code_param_softmax(param_set, sufix):
    string = (
        'int32_t num_rows' + sufix + ' = ' + str(param_set['num_rows']) + ';\n'
        'int32_t row_size' + sufix + ' = ' + str(param_set['row_size']) + ';\n'
        'int32_t mult' + sufix + ' = ' + str(param_set['mult']) + ';\n'
        'int32_t shift' + sufix + ' = ' + str(param_set['shift']) + ';\n'
        'int32_t diff_min' + sufix + ' = ' + str(param_set['diff_min']) + ';\n\n\n'
    )

    return string


def get_c_code_param_add(param_set, sufix):
    string = (
        'int32_t input_1_offset' + sufix + ' = ' + str(param_set['input_1_offset']) + ';\n'
        'int32_t input_1_mult' + sufix + ' = ' + str(param_set['input_1_mult']) + ';\n'
        'int32_t input_1_shift' + sufix + ' = ' + str(param_set['input_1_shift']) + ';\n'
        'int32_t input_2_offset' + sufix + ' = ' + str(param_set['input_2_offset']) + ';\n'
        'int32_t input_2_mult' + sufix + ' = ' + str(param_set['input_2_mult']) + ';\n'
        'int32_t input_2_shift' + sufix + ' = ' + str(param_set['input_2_shift']) + ';\n'
        'int32_t left_shift' + sufix + ' = ' + str(param_set['left_shift']) + ';\n'
        'int32_t out_offset' + sufix + ' = ' + str(param_set['out_offset']) + ';\n'
        'int32_t out_mult' + sufix + ' = ' + str(param_set['out_mult']) + ';\n'
        'int32_t out_shift' + sufix + ' = ' + str(param_set['out_shift']) + ';\n'
        'int32_t out_activation_min' + sufix + ' = ' + str(param_set['out_activation_min']) + ';\n'
        'int32_t out_activation_max' + sufix + ' = ' + str(param_set['out_activation_max']) + ';\n'
        'int32_t block_size' + sufix + ' = ' + str(param_set['block_size']) + ';\n\n'
    )

    return string


def get_c_code_wt_bias_defines():
    string = (
        '#ifndef NN_WT_H \n'
        '#define NN_WT_H \n\n'

        '#include "arm_nn_types.h"\n\n'
    )

    return string


def get_c_code_wt_bias(param_set, sufix):
    weights = np.array2string(np.array(param_set['weights'], dtype=np.int32), separator=',')
    weights = re.sub('[\[\]\n]', '', weights)
    bias = np.array2string(np.array(param_set['bias'], dtype=np.int32), separator=',')
    bias = re.sub('[\[\]\n]', '', bias)

    string = (
        'q7_t wt' + sufix + '[' + str(param_set['weights'].size) + '] = {' + weights + '};\n\n'
        'int32_t bias' + sufix + '[' + str(len(param_set['bias'])) + '] = {' + bias + '};\n\n'
    )

    return string

##############################################################################################
############################### AUXILIAR FUNCTIONS ###########################################
def get_num_layers_with_ctx(cmsis_nn_params):
    layers_with_ctx = ['conv2d', 'batch_normalization', 'average_pooling2d']
    layers_with_ctx_cnt = 0

    for param_set in cmsis_nn_params:
        if any(layer_type in param_set['layer'] for layer_type in layers_with_ctx):
            layers_with_ctx_cnt += 1

    return layers_with_ctx_cnt


def get_ctx_size(cmsis_nn_params, arm_specs):
    ctx_size_list = []

    for param_set in cmsis_nn_params:
        if 'conv2d' in param_set['layer']:
            ctx_size_list.append(get_ctx_size_conv(param_set, arm_specs))
        elif 'batch_normalization' in param_set['layer']:
            ctx_size_list.append(get_ctx_size_dw_conv(param_set, arm_specs))
        elif 'average_pooling2d' in param_set['layer']:
            ctx_size_list.append(get_ctx_size_avg_pool(param_set, arm_specs))

    if len(ctx_size_list) > 0:
        ctx_size = max(ctx_size_list)
    else:
        ctx_size = 0

    return ctx_size


def get_ctx_size_conv(param_set, arm_specs):
    if param_set['conv_params']['padding']['w'] == 0 and param_set['conv_params']['padding']['h'] == 0 and \
    param_set['input_dims']['c'] % 4 == 0 and param_set['conv_params']['stride']['w'] == 1 and \
    param_set['conv_params']['stride']['h'] == 1 and param_set['filter_dims']['w'] == 1 and param_set['filter_dims']['h'] == 1:
        return 0
    elif param_set['output_dims']['h'] == 1 and param_set['input_dims']['h'] == 1 and \
    param_set['filter_dims']['h'] == 1 and param_set['output_dims']['w'] % 4 == 0 and \
    param_set['input_dims']['n'] == 1:
        if arm_specs['arm_math_dsp'] and not arm_specs['arm_math_mvei']:
            return (2 * param_set['input_dims']['c'] * param_set['filter_dims']['w'] * param_set['filter_dims']['h'] * 2)
        else:
            return 0
    else:
        if arm_specs['arm_math_dsp']:
            return (2 * param_set['input_dims']['c'] * param_set['filter_dims']['w'] * param_set['filter_dims']['h'] * 2)
        else:
            return 0


def get_ctx_size_dw_conv(param_set, arm_specs):
    if param_set['input_dims']['c'] == param_set['output_dims']['c'] and param_set['input_dims']['n'] == 1:
        if arm_specs['arm_math_mvei']:
            return (2 * param_set['input_dims']['c'] * param_set['filter_dims']['w'] * param_set['filter_dims']['h'] * 2 + 4)
        elif arm_specs['arm_math_dsp']:
            return (param_set['input_dims']['c'] * param_set['filter_dims']['w'] * param_set['filter_dims']['h'] * 2)
        else:
            return 0
    else:
        return 0


def get_ctx_size_avg_pool(param_set, arm_specs):
    if arm_specs['arm_math_dsp'] and not arm_specs['arm_math_mvei']:
        return (param_set['input_dims']['c'] * 4)
    else:
        return 0
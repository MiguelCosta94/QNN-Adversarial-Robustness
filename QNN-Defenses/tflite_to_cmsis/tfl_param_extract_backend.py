import math
import numpy as np
import tensorflow as tf
import sys


def get_conv_params(input_tensor, filter_tensor, output_tensor, operator):
    input_offset = get_input_offset(input_tensor)
    output_offset = get_output_offset(output_tensor)
    stride = get_stride(operator)
    padding = get_padding_conv(input_tensor, filter_tensor, output_tensor, operator)
    dilation = get_dilation(operator)
    activation_range = get_activation_range(output_tensor)

    cmsis_nn_conv_params = {'input_offset': input_offset, 'output_offset': output_offset,
                            'stride': stride, 'padding': padding, 'dilation': dilation,
                            'activation': activation_range}

    return cmsis_nn_conv_params


def get_dw_conv_params(input_tensor, filter_tensor, output_tensor, operator):
    input_offset = get_input_offset(input_tensor)
    output_offset = get_output_offset(output_tensor)
    ch_mult = operator['builtin_options']['depth_multiplier']
    stride = get_stride(operator)
    padding = get_padding_dw_conv(input_tensor, filter_tensor, output_tensor, operator)
    dilation = get_dilation(operator)
    activation_range = get_activation_range(output_tensor)

    cmsis_nn_dw_conv_params = {'input_offset': input_offset, 'output_offset': output_offset,
                                'ch_mult': ch_mult, 'stride': stride, 'padding': padding,
                                'dilation': dilation, 'activation': activation_range}

    return cmsis_nn_dw_conv_params


def get_fc_params(input_tensor, filter_tensor, output_tensor):
    input_offset = get_input_offset(input_tensor)
    output_offset = get_output_offset(output_tensor)
    filter_offset = -filter_tensor['quantization']['zero_point'][0]
    activation_range = get_activation_range(output_tensor)

    fc_params = {'input_offset': input_offset, 'output_offset': output_offset,
                'filter_offset': filter_offset, 'activation': activation_range}

    return fc_params


def get_avg_pool_params(input_tensor, filter_tensor, output_tensor, operator):
    stride = get_stride(operator)
    padding = get_padding_avg_pool(input_tensor, filter_tensor, output_tensor, operator)
    activation_range = get_activation_range(output_tensor)

    cmsis_nn_avg_pool_params = {'stride': stride, 'padding': padding, 'activation': activation_range}

    return cmsis_nn_avg_pool_params


def get_input_offset(input_tensor):
    input_offset = -input_tensor['quantization']['zero_point'][0]
    return input_offset


def get_output_offset(output_tensor):
    output_offset = output_tensor['quantization']['zero_point'][0]
    return output_offset


def get_quant_params(input_tensor, filter_tensor, output_tensor):
    mantissa_fixed_per_ch = []
    shift_per_ch = []

    input_scale = input_tensor['quantization']['scale']
    filter_scale = filter_tensor['quantization']['scale']
    output_scale = output_tensor['quantization']['scale']

    effective_output_scale_per_ch = np.multiply(input_scale, filter_scale)
    effective_output_scale_per_ch = np.divide(effective_output_scale_per_ch, output_scale)

    for effective_output_scale in effective_output_scale_per_ch:
        mantissa_fixed, shift = get_mantissa_shift(effective_output_scale)
        mantissa_fixed = np.array(mantissa_fixed, dtype=np.int32)
        mantissa_fixed_per_ch.append(mantissa_fixed)
        shift_per_ch.append(shift)

    return {'multiplier': mantissa_fixed_per_ch, 'shift': shift_per_ch}


def get_softmax_quant_params(input_tensor, operator):
    input_integer_bits = 5
    total_signed_bits = 31
    beta = operator['builtin_options']['beta']
    input_scale = input_tensor['quantization']['scale'][0]

    input_beta_real_multiplier = beta * input_scale * (1 << (31 - input_integer_bits))
    input_beta_real_multiplier = np.minimum(input_beta_real_multiplier, (1 << 31) - 1.0)

    mantissa_fixed, shift = get_mantissa_shift(input_beta_real_multiplier)

    max_input_rescaled = ((1 << input_integer_bits) - 1) * (1 << (total_signed_bits - input_integer_bits)) / (1 << shift)
    max_input_rescaled = math.floor(max_input_rescaled)
    diff_min = -1 * max_input_rescaled

    return mantissa_fixed, shift, diff_min


def get_add_quant_params(input_tensors, output_tensor, left_shift):
    i1_scale = input_tensors[0]['quantization']['scale'][0]
    i2_scale = input_tensors[1]['quantization']['scale'][0]
    o_scale = output_tensor['quantization']['scale'][0]
    lst = [i1_scale, i2_scale]
    twice_max_input_scale = 2 * max(lst)
    i1_real_multiplier = i1_scale / twice_max_input_scale
    i2_real_multiplier = i2_scale / twice_max_input_scale
    o_real_multiplier = twice_max_input_scale / ((1 << left_shift) * o_scale)

    i1_mantissa_fixed, i1_shift = get_mantissa_shift(i1_real_multiplier)
    i2_mantissa_fixed, i2_shift = get_mantissa_shift(i2_real_multiplier)
    o_mantissa_fixed, o_shift = get_mantissa_shift(o_real_multiplier)

    i1_quant_params = {'multiplier': i1_mantissa_fixed, 'shift': i1_shift}
    i2_quant_params = {'multiplier': i2_mantissa_fixed, 'shift': i2_shift}
    o_quant_params = {'multiplier': o_mantissa_fixed, 'shift': o_shift}

    return i1_quant_params, i2_quant_params, o_quant_params


def get_tensor_dims(tensor):
    if len(tensor['shape']) == 4:
        return get_4d_tensor_dims(tensor)
    elif len(tensor['shape']) == 2:
        return get_2d_tensor_dims(tensor)
    else:
        print("ERROR: get_tensor_dims onyl support 4D and 2D tensors")
        sys.exit(1)


def get_tensor_size(tensor):
    size = 1
    
    for dim in tensor['shape']:
        size *= dim

    return size


def get_input_dims_softmax(input_tensor):
    try:
        num_rows = input_tensor['shape'][0]
        row_size = input_tensor['shape'][1]
    except:
        print('ERROR: get_input_dims_softmax only supports tensors with at least 2 dimensions')
        sys.exit(1)

    return num_rows, row_size


def get_output_dims_fc(output_tensor):
    if len(output_tensor['shape']) >= 2:
        dims = {'n': output_tensor['shape'][0], 'h': 1, 'w': 1, 'c': output_tensor['shape'][len(output_tensor['shape'])-1]}
    else:
        print('ERROR: get_output_dims_fc only supports 4D tensors')
        sys.exit(1)

    return dims


def get_filter_dims_conv(input_tensor, filter_tensor, output_tensor):
    try:
        dim = {'n': get_tensor_dims(output_tensor)['c'],
                'h': filter_tensor['shape'][1],
                'w': filter_tensor['shape'][2],
                'c': get_tensor_dims(input_tensor)['c']}
    except:
        print('ERROR: get_filter_dims_conv only supports 4D tensors')
        sys.exit(1)

    return dim


def get_filter_dims_dw_conv(filter_tensor, output_tensor):
    try:
        dim = {'n': 1,
                'h': filter_tensor['shape'][1],
                'w': filter_tensor['shape'][2],
                'c': get_tensor_dims(output_tensor)['c']}
    except:
        print('ERROR: get_filter_dims_dw_conv only supports 4D tensors')
        sys.exit(1)

    return dim


def get_filter_dims_fc(input_tensor, output_tensor):
    input_dims = get_tensor_dims(input_tensor)
    n = input_dims['h'] * input_dims['w'] * input_dims['c']
    dims = {'n': n, 'h': 1, 'w': 1, 'c': get_output_dims_fc(output_tensor)['c']}

    return dims


def get_filter_dims_avg_pool(operator):
    filter_dims = {'n': 1, 'h': operator['builtin_options']['filter_height'],
                    'w': operator['builtin_options']['filter_width'], 'c': 1}

    return filter_dims


def get_bias_dims(bias_tensor):
    dim = {'n': 1, 'h': 1, 'w': 1, 'c': bias_tensor['shape'][0]}

    return dim


def get_weights(filter_tensor, buffers):
    weights = buffers[filter_tensor['buffer']]['data']
    weights = tf.cast(weights, tf.int8).numpy()

    return weights


def get_bias(bias_tensor, buffers):
    bias_byte_array = buffers[bias_tensor['buffer']]['data']
    size_buffer = len(bias_byte_array)
    index_buffer = 0
    bias_list = []

    while index_buffer < size_buffer:
        byte_array = bias_byte_array[index_buffer: index_buffer+4]
        data_int32 = int.from_bytes(byte_array, byteorder='little', signed=True)
        bias_list.append(data_int32)
        index_buffer += 4
    
    return bias_list
    
##################################################################################
############################## AUXILIAR FUNCTIONS ################################
def get_stride(operator):
    stride = {'h': operator['builtin_options']['stride_h'],
                'w': operator['builtin_options']['stride_w']}

    return stride


def get_padding_conv(input_tensor, filter_tensor, output_tensor, operator):
    stride_h = get_stride(operator)['h']
    dilation_h = get_dilation(operator)['h']
    input_dim_h = get_tensor_dims(input_tensor)['h']
    filter_dim_h = get_filter_dims_conv(input_tensor, filter_tensor, output_tensor)['h']
    out_dim_h = get_tensor_dims(output_tensor)['h']
    padding_h = calculate_padding(stride_h, dilation_h, input_dim_h, filter_dim_h, out_dim_h)

    stride_w = get_stride(operator)['w']
    dilation_w = get_dilation(operator)['w']
    input_dim_w = get_tensor_dims(input_tensor)['w']
    filter_dim_w = get_filter_dims_conv(input_tensor, filter_tensor, output_tensor)['w']
    out_dim_w = get_tensor_dims(output_tensor)['w']
    padding_w = calculate_padding(stride_w, dilation_w, input_dim_w, filter_dim_w, out_dim_w)

    padding = {'h': padding_h, 'w': padding_w}

    return padding


def get_padding_dw_conv(input_tensor, filter_tensor, output_tensor, operator):
    stride_h = get_stride(operator)['h']
    dilation_h = get_dilation(operator)['h']
    input_dim_h = get_tensor_dims(input_tensor)['h']
    filter_dim_h = get_filter_dims_dw_conv(filter_tensor, output_tensor)['h']
    out_dim_h = get_tensor_dims(output_tensor)['h']
    padding_h = calculate_padding(stride_h, dilation_h, input_dim_h, filter_dim_h, out_dim_h)

    stride_w = get_stride(operator)['w']
    dilation_w = get_dilation(operator)['w']
    input_dim_w = get_tensor_dims(input_tensor)['w']
    filter_dim_w = get_filter_dims_dw_conv(filter_tensor, output_tensor)['w']
    out_dim_w = get_tensor_dims(output_tensor)['w']
    padding_w = calculate_padding(stride_w, dilation_w, input_dim_w, filter_dim_w, out_dim_w)

    padding = {'h': padding_h, 'w': padding_w}

    return padding


def get_padding_avg_pool(input_tensor, filter_tensor, output_tensor, operator):
    stride_h = get_stride(operator)['h']
    dilation_h = 1
    input_dim_h = get_tensor_dims(input_tensor)['h']
    filter_dim_h = get_filter_dims_avg_pool(operator)['h']
    out_dim_h = get_tensor_dims(output_tensor)['h']
    padding_h = calculate_padding(stride_h, dilation_h, input_dim_h, filter_dim_h, out_dim_h)

    stride_w = get_stride(operator)['w']
    dilation_w = 1
    input_dim_w = get_tensor_dims(input_tensor)['w']
    filter_dim_w = get_filter_dims_avg_pool(operator)['w']
    out_dim_w = get_tensor_dims(output_tensor)['w']
    padding_w = calculate_padding(stride_w, dilation_w, input_dim_w, filter_dim_w, out_dim_w)

    padding = {'h': padding_h, 'w': padding_w}

    return padding


def calculate_padding(stride, dilation_rate, in_size, filter_size, out_size):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = (out_size - 1) * stride + effective_filter_size - in_size

    if total_padding <= 0:
        total_padding = 0

    total_padding = int(total_padding)
    offset = total_padding % 2
   
    return int(total_padding / 2)


def get_dilation(operator):
    dilation = {'h': operator['builtin_options']['dilation_h_factor'],
                'w': operator['builtin_options']['dilation_w_factor']}

    return dilation


def get_activation_range(tensor):
    activation_range = {'min': -128, 'max':127}

    if 'Relu' in tensor['name']:
        activation_range['min'] = max(activation_range['min'], tensor['quantization']['zero_point'][0])

    return activation_range


def get_mantissa_shift(scale):
    mantissa, shift = math.frexp(scale)
    mantissa_fixed = mantissa * (1 << 31)
    mantissa_fixed = round(mantissa_fixed)

    if mantissa_fixed == (1 << 31):
        mantissa_fixed = mantissa_fixed / 2
        shift = shift + 1

    if shift < -31:
        shift = 0
        mantissa_fixed = 0

    return mantissa_fixed, shift


def round(number_to_round):
    diff = number_to_round - int(number_to_round)
    if diff >= 0.5:
        rounded_number = int(number_to_round) + 1
    else:
        rounded_number = int(number_to_round)

    return rounded_number


def get_4d_tensor_dims(tensor):
    dim = {'n': 1, 'h': 1, 'w': 1, 'c': 1}
    keys = list(dim.keys())
    dim_cnt = 0

    while dim_cnt < len(tensor['shape']):
        dim[keys[dim_cnt]] = tensor['shape'][dim_cnt]
        dim_cnt = dim_cnt + 1

    return dim


def get_2d_tensor_dims(tensor):
    dim = {'n': 1, 'h': 1, 'w': 1, 'c': 1}
    keys = list(dim.keys())
    dim_cnt = 0

    while dim_cnt < len(tensor['shape']):
        dim[keys[dim_cnt+1]] = tensor['shape'][dim_cnt]
        dim_cnt = dim_cnt + 1

    return dim
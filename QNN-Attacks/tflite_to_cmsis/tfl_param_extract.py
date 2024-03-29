from tflite_to_cmsis import tfl_param_extract_backend as peb
from tflite_to_cmsis import tfl_reader as tflr


def get_cmsis_nn_params(tensors, operators, buffers):
    cmsis_nn_params = []

    for operator in operators:
        input_tensors, filter_tensor, bias_tensor, output_tensor = tflr.get_operator_tensors(operator, tensors)
        if ('add' in filter_tensor['name']) and ('Relu' in filter_tensor['name']):
            filter_name = filter_tensor['name'].split(';')[1]
        else:
            filter_name = filter_tensor['name'].split(';')[0]

        if 'conv2d' in filter_name:
            input_tensor = input_tensors[0]
            layer_cmsis_params = get_conv_layer_params(input_tensor, filter_tensor, bias_tensor,
                                                            output_tensor, operator, buffers)
            cmsis_nn_params.append(layer_cmsis_params)

        elif 'batch_normalization' in filter_name:
            input_tensor = input_tensors[0]
            layer_cmsis_params = get_dw_conv_layer_params(input_tensor, filter_tensor,
                                                            bias_tensor, output_tensor, operator, buffers)
            cmsis_nn_params.append(layer_cmsis_params)

        elif 'average_pooling2d' in filter_name:
            input_tensor = input_tensors[0]
            layer_cmsis_params = get_avg_pool_layer_params(input_tensor, filter_tensor,
                                                            output_tensor, operator)
            cmsis_nn_params.append(layer_cmsis_params)

        elif 'dense' in filter_name:
            input_tensor = input_tensors[0]
            layer_cmsis_params = get_fc_layer_params(input_tensor, filter_tensor,
                                                            bias_tensor, output_tensor, buffers)
            cmsis_nn_params.append(layer_cmsis_params)

        elif 'Identity' in filter_name:
            input_tensor = input_tensors[0]
            layer_cmsis_params = get_softmax_params(input_tensor, filter_tensor, operator)
            cmsis_nn_params.append(layer_cmsis_params)

        elif 'add' in filter_tensor['name']:
            layer_cmsis_params = get_add_params(input_tensors, filter_tensor, output_tensor)
            cmsis_nn_params.append(layer_cmsis_params)

    return cmsis_nn_params


def get_conv_layer_params(input_tensor, filter_tensor, bias_tensor, output_tensor, operator, buffers):
    conv_params = peb.get_conv_params(input_tensor, filter_tensor, output_tensor, operator)
    quant_params = peb.get_quant_params(input_tensor, filter_tensor, output_tensor)
    input_dims = peb.get_tensor_dims(input_tensor)
    output_dims = peb.get_tensor_dims(output_tensor)
    filter_dims = peb.get_filter_dims_conv(input_tensor, filter_tensor, output_tensor)
    bias_dims = peb.get_bias_dims(bias_tensor)
    weights = peb.get_weights(filter_tensor, buffers)
    bias = peb.get_bias(bias_tensor, buffers)
    output_size = peb.get_tensor_size(output_tensor)

    layer_cmsis_params = {'layer': filter_tensor['name'].split(';')[0], 'conv_params': conv_params, 'quant_params': quant_params,
                        'input_dims': input_dims, 'filter_dims': filter_dims, 'bias_dims': bias_dims, 'output_dims': output_dims,
                        'weights': weights, 'bias': bias, 'output_size': output_size}

    return layer_cmsis_params


def get_dw_conv_layer_params(input_tensor, filter_tensor, bias_tensor, output_tensor, operator, buffers):
    dw_conv_params = peb.get_dw_conv_params(input_tensor, filter_tensor, output_tensor, operator)
    quant_params = peb.get_quant_params(input_tensor, filter_tensor, output_tensor)
    input_dims = peb.get_tensor_dims(input_tensor)
    output_dims = peb.get_tensor_dims(output_tensor)
    filter_dims = peb.get_filter_dims_dw_conv(filter_tensor, output_tensor)
    bias_dims = peb.get_bias_dims(bias_tensor)
    weights = peb.get_weights(filter_tensor, buffers)
    bias = peb.get_bias(bias_tensor, buffers)

    layer_cmsis_params = {'layer': filter_tensor['name'].split(';')[0], 'dw_conv_params': dw_conv_params,
                            'quant_params': quant_params, 'input_dims': input_dims, 'filter_dims': filter_dims,
                            'bias_dims': bias_dims, 'output_dims': output_dims, 'weights': weights, 'bias': bias}

    return layer_cmsis_params


def get_fc_layer_params(input_tensor, filter_tensor, bias_tensor, output_tensor, buffers):
    fc_params = peb.get_fc_params(input_tensor, filter_tensor, output_tensor)
    quant_params = peb.get_quant_params(input_tensor, filter_tensor, output_tensor)
    input_dims = peb.get_tensor_dims(input_tensor)
    output_dims = peb.get_output_dims_fc(output_tensor)
    filter_dims = peb.get_filter_dims_fc(input_tensor, output_tensor)
    bias_dims = peb.get_bias_dims(bias_tensor)
    weights = peb.get_weights(filter_tensor, buffers)
    bias = peb.get_bias(bias_tensor, buffers)

    layer_cmsis_params = {'layer': filter_tensor['name'].split(';')[0], 'fc_params': fc_params,
                            'quant_params': quant_params, 'input_dims': input_dims, 'filter_dims': filter_dims,
                            'bias_dims': bias_dims, 'output_dims': output_dims, 'weights': weights, 'bias': bias}

    return layer_cmsis_params


def get_avg_pool_layer_params(input_tensor, filter_tensor, output_tensor, operator):
    pool_params = peb.get_avg_pool_params(input_tensor, filter_tensor, output_tensor, operator)
    input_dims = peb.get_tensor_dims(input_tensor)
    output_dims = peb.get_tensor_dims(output_tensor)
    filter_dims = peb.get_filter_dims_avg_pool(operator)

    layer_cmsis_params = {'layer': filter_tensor['name'].split(';')[0], 'pool_params': pool_params,
                            'input_dims': input_dims, 'filter_dims': filter_dims, 'output_dims': output_dims}

    return layer_cmsis_params


def get_softmax_params(input_tensor, filter_tensor, operator):
    num_rows, row_size = peb.get_input_dims_softmax(input_tensor)
    mult, shift, diff_min = peb.get_softmax_quant_params(input_tensor, operator)

    layer_cmsis_params = {'layer': filter_tensor['name'].split(';')[0], 'num_rows': num_rows,
                            'row_size': row_size, 'mult': mult, 'shift': shift,
                            'diff_min': diff_min}

    return layer_cmsis_params


def get_add_params(input_tensors, filter_tensor, output_tensor):
    input_1_offset = peb.get_input_offset(input_tensors[0])
    input_2_offset = peb.get_input_offset(input_tensors[1])
    out_offset = peb.get_output_offset(output_tensor)
    left_shift = 20
    i1_quant_params, i2_quant_params, o_quant_params = peb.get_add_quant_params(input_tensors, output_tensor, left_shift)
    activation_range = peb.get_activation_range(output_tensor)
    block_size = peb.get_tensor_size(output_tensor)

    layer_cmsis_params = {'layer': filter_tensor['name'], 'input_1_offset': input_1_offset,
        'input_1_mult': i1_quant_params['multiplier'], 'input_1_shift': i1_quant_params['shift'],
        'input_2_offset': input_2_offset, 'input_2_mult': i2_quant_params['multiplier'],
        'input_2_shift': i2_quant_params['shift'], 'left_shift': left_shift, 'out_offset': out_offset,
        'out_mult': o_quant_params['multiplier'], 'out_shift': o_quant_params['shift'],
        'out_activation_min': activation_range['min'], 'out_activation_max': activation_range['max'], 'block_size': block_size}

    return layer_cmsis_params
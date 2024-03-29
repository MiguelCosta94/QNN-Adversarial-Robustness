from tflite_to_cmsis import tfl_reader as tflr
import numpy as np


def get_buf_io_map_resnet(operators, tensors):
    valid_layers = ['conv2d', 'average_pooling2d', 'dense', 'Identity', 'add']
    idxs_div_layers = tflr.get_idxs_div_layers(operators, tensors)
    locked_buffer = None
    cmsis_buf_map = []
    
    for idx_op, operator in enumerate(operators):
        input = []
        output = []
        input_tensors, filter_tensor, bias_tensor, output_tensor = tflr.get_operator_tensors(operator, tensors)
        if ('add' in filter_tensor['name']) and ('Relu' in filter_tensor['name']):
            filter_name = filter_tensor['name'].split(';')[1]
        else:
            filter_name = filter_tensor['name'].split(';')[0]

        if any(layer_type in filter_name for layer_type in valid_layers):
            if idx_op == 0:
                input.append(0)
                if idx_op in idxs_div_layers:
                    output.append(1)
                    output.append(2)
                    locked_buffer = 2
                else:
                    output.append(1)
            else:
                idx_op_i1 = tflr.get_operator_idx_from_output_tensor(input_tensors[0], tensors, operators)

                if len(input_tensors) == 1:
                    if idx_op_i1 == idx_op - 1:
                        i1 = cmsis_buf_map[len(cmsis_buf_map)-1]['output'][0]
                    else:
                        i1 = locked_buffer
                        locked_buffer = cmsis_buf_map[len(cmsis_buf_map)-1]['output'][0]
                    input.append(i1)

                    if idx_op in idxs_div_layers:
                        o1 = get_output_buffer_resnet([i1, locked_buffer])
                        o2 = get_output_buffer_resnet([o1, locked_buffer])
                        output.append(o1)
                        output.append(o2)
                        locked_buffer = o2
                    else:
                        o1 = get_output_buffer_resnet([i1, locked_buffer])
                        output.append(o1)

                elif len(input_tensors) == 2:
                    if idx_op_i1 == idx_op - 1:
                        i1 = cmsis_buf_map[len(cmsis_buf_map)-1]['output'][0]
                        i2 = locked_buffer
                    else:
                        i1 = locked_buffer
                        i2 = cmsis_buf_map[len(cmsis_buf_map)-1]['output'][0]
                    input.append(i1)
                    input.append(i2)
                    locked_buffer = None

                    if idx_op in idxs_div_layers:
                        o1 = get_output_buffer_resnet([i1, i2, locked_buffer])
                        o2 = get_output_buffer_resnet([o1, locked_buffer])
                        output.append(o1)
                        output.append(o2)
                        locked_buffer = o2
                    else:
                        o1 = get_output_buffer_resnet([i1, locked_buffer])
                        output.append(o1)

            entry = {'layer': filter_name, 'input': input, 'output': output}
            cmsis_buf_map.append(entry)
    
    return cmsis_buf_map


def get_output_buffer_resnet(locked_buffers):
    valid_buffers = [0,1,2]
    effective_locked_buffers = [i for i in locked_buffers if i!=None]
    free_buffers = np.setdiff1d(valid_buffers, effective_locked_buffers)

    if len(free_buffers) == 0:
        print("ERROR! NO FREE BUFFERS TO ALLOCATE THE OUTPUT")
        sys.exit(0)
    else:
        return free_buffers[0]


def get_buf_io_map_std(operators, tensors):
    valid_layers = ['conv2d', 'batch_normalization', 'average_pooling2d', 'max_pooling2d', 'dense', 'Identity', 'add']
    cmsis_buf_map = []
    buffer_idx = 0

    for operator in operators:
        input_tensors, filter_tensor, bias_tensor, output_tensor = tflr.get_operator_tensors(operator, tensors)

        if ('add' in filter_tensor['name']) and ('Relu' in filter_tensor['name']):
            filter_name = filter_tensor['name'].split(';')[1]
        else:
            filter_name = filter_tensor['name'].split(';')[0]

        if any(layer_type in filter_name for layer_type in valid_layers):
            entry = {'layer': filter_name, 'input': [buffer_idx], 'output': [buffer_idx^1]}
            buffer_idx = buffer_idx^1
            cmsis_buf_map.append(entry)

    return cmsis_buf_map
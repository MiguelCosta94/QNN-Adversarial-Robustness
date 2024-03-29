import re
import sys
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb


def CreateDictFromFlatbuffer(buffer_data):
    model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)

    return FlatbufferToDict(model, preserve_as_numpy=False)


def NameListToString(name_list):
    """Converts a list of integers to the equivalent ASCII string."""
    if isinstance(name_list, str):
        return name_list
    else:
        result = ""
        if name_list is not None:
            for val in name_list:
                result = result + chr(int(val))
        return result


def tensors_name_to_ascii(tensor_list, model_id):
    for tensor in tensor_list:
        tensor['name'] = NameListToString(tensor['name'])

    if model_id == 'cif' or model_id == 'kws':
        tensor_list[-1]['name'] = 'Identity_int8'

    return tensor_list


def get_operator_tensors(operator, tensors):
    input_tensors = []
    output_tensor_index = operator['outputs'][0]
    output_tensor = tensors[output_tensor_index]

    # Conv, DW-Conv, and FC layers in TFLite models has always bias
    if len(operator['inputs']) == 3:
        filter_tensor_index = operator['inputs'][1]
        filter_tensor = tensors[filter_tensor_index]
        bias_tensor_index = operator['inputs'][2]
        bias_tensor = tensors[bias_tensor_index]
    else:
        filter_tensor = output_tensor
        bias_tensor = None

    # Relu layers in ResNets can have two input tensors
    if (len(operator['inputs']) == 2) and ('add' in filter_tensor['name']):
        input_tensor_index = operator['inputs'][0]
        input_tensors.append(tensors[input_tensor_index])
        input_tensor_index = operator['inputs'][1]
        input_tensors.append(tensors[input_tensor_index])
    else:
        input_tensor_index = operator['inputs'][0]
        input_tensors.append(tensors[input_tensor_index])


    return input_tensors, filter_tensor, bias_tensor, output_tensor


def get_operator_idx_from_output_tensor(ref_out_tensor, tensors, operators):
    for idx, operator in enumerate(operators):
        input_tensors, filter_tensor, bias_tensor, output_tensor = get_operator_tensors(operator, tensors)
        if output_tensor == ref_out_tensor:
            return idx

    print("ERROR! THERE IS NO OPERATOR WITH THE GIVEN OUTPUT TENSOR")
    sys.exit(0)


def get_idxs_div_layers(operators, tensors):
    idx_layer_div_list = []

    for idx_op_1, operator in enumerate(operators):
        input_tensors_1, filter_tensor_1, bias_tensor_1, output_tensor_1 = get_operator_tensors(operator, tensors)
        idx_op_2 = idx_op_1 + 1
        is_already_input = False

        while idx_op_2 < len(operators):
            input_tensors_2, filter_tensor_2, bias_tensor_2, output_tensor_2 = get_operator_tensors(operators[idx_op_2], tensors)
            if output_tensor_1 in input_tensors_2:
                if is_already_input == False:
                    is_already_input = True
                else:
                    idx_layer_div_list.append(idx_op_1)
                    break
            idx_op_2 += 1

    return idx_layer_div_list


def get_idx_layers_break_fw(operators, tensors):
    idx_layers_break_fw = []

    for idx_op, operator in enumerate(operators):
        if idx_op > 0:
            input_tensors_curr, t2, t3, t4 = get_operator_tensors(operator, tensors)
            t1, t2, t3, output_tensor_prev = get_operator_tensors(operators[idx_op-1], tensors)

            if output_tensor_prev not in input_tensors_curr:
                idx_layers_break_fw.append(idx_op)

    return idx_layers_break_fw


def get_idx_add_layers(operators, tensors):
    idx_add_layers = []

    for idx_op, operator in enumerate(operators):
        input_tensors, filter_tensor, bias_tensor, output_tensor = get_operator_tensors(operator, tensors)
        filter_name = filter_tensor['name']

        if ('Relu' in filter_name) and ('add' in filter_name):
            idx_add_layers.append(idx_op)

    return idx_add_layers 
###################################################
#############AUXILIARY FUNCTIONS###################
def FlatbufferToDict(fb, preserve_as_numpy):
    """Converts a hierarchy of FB objects into a nested dict.

    We avoid transforming big parts of the flat buffer into python arrays. This
    speeds conversion from ten minutes to a few seconds on big graphs.

    Args:
      fb: a flat buffer structure. (i.e. ModelT)
      preserve_as_numpy: true if all downstream np.arrays should be preserved.
        false if all downstream np.array should become python arrays
    Returns:
      A dictionary representing the flatbuffer rather than a flatbuffer object.
    """
    if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
        return fb
    elif hasattr(fb, "__dict__"):
        result = {}
        for attribute_name in dir(fb):
            attribute = fb.__getattribute__(attribute_name)
            if not callable(attribute) and attribute_name[0] != "_":
                snake_name = CamelCaseToSnakeCase(attribute_name)
                preserve = True if attribute_name == "buffers" else preserve_as_numpy
                result[snake_name] = FlatbufferToDict(attribute, preserve)
        return result
    elif isinstance(fb, np.ndarray):
        return fb if preserve_as_numpy else fb.tolist()
    elif hasattr(fb, "__len__"):
        return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
    else:
        return fb


def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
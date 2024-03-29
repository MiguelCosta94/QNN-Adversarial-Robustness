def get_c_code_main_function_usb(buf_size, model_id):
    string = (
        '#include "File_Handling.h"\n'
        '#include "arm_nn_types.h"\n' 
        '#include "arm_nnfunctions.h"\n'
        '#include "nn_wt.h"\n'
        '#include "nn_params.h"\n\n'

        'q7_t buffer0[' + str(buf_size) + '];\n'
        'q7_t buffer1[' + str(buf_size) + '];\n'
    )

    if model_id == 'cif':
        string += (
            'q7_t buffer2[' + str(buf_size) + '];\n'
        )

    if model_id == 'ano':
        string += (
            'float ann_in_f[INPUT_DATA_SIZE];\n'
        )
    
    string += (
        'q7_t buffer_ctx[CTX_SIZE];\n'
        'FIL USBHDataFile;\n'
        'FIL USBHLabelsFile;\n\n'

        'int check_ctx_size();\n'
    )

    if model_id == 'ano':
        string += (
        'void dequantize(q7_t *input, float scale, float zero_point, uint16_t input_length, float *output);\n'
        'float run_ann(q7_t *input);\n\n'
        )
    else:
        string += (
            'q7_t run_ann(q7_t *input);\n\n'
        )

    string += (
        'int main()\n' 
        '{\n'
            '\tFRESULT status_data_file;\n'
    )

    if model_id == 'ano':
        string += (
            '\tunion {\n'
                '\t\tfloat val;\n'
                '\t\tuint8_t bytes[4];\n'
            '\t} ann_out;\n\n'
        )
    else:
        string += (
            '\tq7_t ann_out;\n'
        )
            
    string += (
        '\tif(!check_ctx_size()){\n'
            '\t\treturn 0;\n'
        '\t}\n\n'
                                    
        '\twhile(get_usb_status() == 0){\n'
        '\t\tMX_USB_HOST_Process();\n'
        '\t}\n\n'
                            
        '\tstatus_data_file = Open_File("data.bin", &USBHDataFile);\n'
                                                        
        '\tif(status_data_file != FR_OK){\n'
        '\t\treturn 0;\n'
        '\t}\n\n'

        '\tfor(uint16_t i=0; i<DATASET_SIZE; i++) {\n'
            '\t\tfor(uint16_t j=0; j<BATCH_SIZE; j++) {\n'
                '\t\t\tstatus_data_file = Read_File_Batch("data.bin", &USBHDataFile, INPUT_DATA_SIZE, (char*)buffer0);\n'
                                                        
                '\t\t\tif(status_data_file != FR_OK){\n'
                '\t\t\t\tbreak;\n'
                '\t\t\t}\n\n'
    )

    if model_id == 'ano':
        string += (
            '\t\t\tann_out.val = run_ann(buffer0);\n'
            '\t\t\tif(HAL_UART_Transmit(&hlpuart1, ann_out.bytes, 4, 0xFFFFFFFF) != HAL_OK) {\n'
        )
    else:
        string += (
            '\t\t\tann_out = run_ann(buffer0);\n'
            '\t\t\tif(HAL_UART_Transmit(&hlpuart1, (uint8_t*)&ann_out, 1, 0xFFFFFFFF) != HAL_OK) {\n'
        )

    string += (
                    '\t\t\t\t//acender led\n'
                '\t\t\t}\n'
            '\t\t}\n'
        '\t}\n\n'

        '\treturn 1;\n'
        '}\n\n'
    )

    return string


def get_c_code_main_function_ser(buf_size, model_id):
    string = (
        '#include "arm_nn_types.h"\n' 
        '#include "arm_nnfunctions.h"\n'
        '#include "nn_wt.h"\n'
        '#include "nn_params.h"\n\n'

        'q7_t buffer0[' + str(buf_size) + '];\n'
        'q7_t buffer1[' + str(buf_size) + '];\n'
    )

    if model_id == 'cif':
        string += (
            'q7_t buffer2[' + str(buf_size) + '];\n'
        )

    if model_id == 'ano':
        string += (
            'float ann_in_f[INPUT_DATA_SIZE];\n'
        )
    
    string += (
        'q7_t buffer_ctx[CTX_SIZE];\n'

        'int check_ctx_size();\n'
    )

    if model_id == 'ano':
        string += (
        'void dequantize(q7_t *input, float scale, float zero_point, uint16_t input_length, float *output);\n'
        'float run_ann(q7_t *input);\n\n'
        )
    else:
        string += (
            'q7_t run_ann(q7_t *input);\n\n'
        )

    string += (
        'int main()\n' 
        '{\n'
    )

    if model_id == 'ano':
        string += (
            '\tunion {\n'
                '\t\tfloat val;\n'
                '\t\tuint8_t bytes[4];\n'
            '\t} ann_out;\n\n'
        )
    else:
        string += (
            '\tq7_t ann_out;\n'
        )
            
    string += (
        '\tif(!check_ctx_size()){\n'
            '\t\treturn 0;\n'
        '\t}\n\n'

        '\tfor(uint16_t i=0; i<DATASET_SIZE; i++) {\n'
            '\t\tfor(uint16_t j=0; j<BATCH_SIZE; j++) {\n'
                '\t\t\tHAL_UART_Receive(&hlpuart1, (uint8_t*)buffer0, INPUT_DATA_SIZE, 0xFFFFFF);\n\n'
    )

    if model_id == 'ano':
        string += (
            '\t\t\tann_out.val = run_ann(buffer0);\n'
            '\t\t\tif(HAL_UART_Transmit(&hlpuart1, ann_out.bytes, 4, 0xFFFFFFFF) != HAL_OK) {\n'
        )
    else:
        string += (
            '\t\t\tann_out = run_ann(buffer0);\n'
            '\t\t\tif(HAL_UART_Transmit(&hlpuart1, (uint8_t*)&ann_out, 1, 0xFFFFFFFF) != HAL_OK) {\n'
        )

    string += (
                    '\t\t\t\t//acender led\n'
                '\t\t\t}\n'
            '\t\t}\n'
        '\t}\n\n'

        '\treturn 1;\n'
        '}\n\n'
    )

    return string


def get_c_code_check_ctx_size(cmsis_nn_params):
    conv_cnt = 0
    dw_conv_cnt = 0
    avg_pool_cnt = 0
    fc_cnt = 0

    file_string = (
        'int check_ctx_size()\n'
        '{\n'
            '\tint32_t buffer_tmp[NUM_LAYERS_WITH_CTX];\n'
            '\tint32_t buffer_aux_max_size = 0;\n\n'
    )

    for param_set in cmsis_nn_params:
        buffer_tmp_index = conv_cnt + dw_conv_cnt + avg_pool_cnt + fc_cnt
        var_name = 'buffer_tmp[' + str(buffer_tmp_index) + ']'

        if 'conv2d' in param_set['layer']:
            code_string = get_c_code_ctx_size_conv('_conv' + str(conv_cnt), var_name)
            file_string += code_string
            conv_cnt += 1
        elif 'batch_normalization' in param_set['layer']:
            code_string = get_c_code_ctx_size_dw_conv('_dw_cnv' + str(dw_conv_cnt), var_name)
            file_string += code_string
            dw_conv_cnt += 1
        elif 'average_pooling2d' in param_set['layer']:
            code_string = get_c_code_ctx_size_avg_pool('_avg_pool' + str(avg_pool_cnt), var_name)
            file_string += code_string
            avg_pool_cnt += 1

    file_string += (
        '\n\tfor(int i=0; i<NUM_LAYERS_WITH_CTX; i++) {\n'
            '\t\tif(buffer_tmp[i] > CTX_SIZE) {\n'
                '\t\t\treturn 0;\n'
            '\t\t}\n'
        '\t}\n\n'

        '\treturn 1;\n'
        '}\n\n'
    )

    return file_string


def get_c_code_dequantize():
    string = (
        'void dequantize(q7_t *input, float scale, float zero_point, uint16_t input_length, float *output)\n'
        '{\n'
            '\tfor(uint16_t i=0; i<input_length; i++) {\n'
            '\t\toutput[i] = (input[i] * scale) - (zero_point * scale);\n'
            '\t}\n'
        '}\n\n'
    )

    return string


def get_c_code_run_ann(cmsis_nn_params, cmsis_nn_buffers_map, model_id):
    conv_cnt = 0
    dw_conv_cnt = 0
    avg_pool_cnt = 0
    fc_cnt = 0
    softmax_cnt = 0
    add_cnt = 0

    if model_id == 'ano':
        file_string = (
            'float run_ann(q7_t *input_data)\n'
            '{\n'
                '\tcmsis_nn_context ctx = {.buf=buffer_ctx, .size=CTX_SIZE};\n'
                '\tfloat prediction = 0;\n'
                '\tfloat ann_out_f;\n'
                '\tfloat error;\n\n'

                '\tdequantize(buffer0, ANN_IN_QUANT_SCALE, ANN_IN_QUANT_ZP, INPUT_DATA_SIZE, ann_in_f);\n\n'
        )
    else:
        file_string = (
            'q7_t run_ann(q7_t *input_data)\n'
            '{\n'
                '\tcmsis_nn_context ctx = {.buf=buffer_ctx, .size=CTX_SIZE};\n'
                '\tq7_t prediction = 0;\n\n'
        )

    for i, param_set in enumerate(cmsis_nn_params):
        is_first_layer = False
        if i == 0:
            is_first_layer = True
        
        if 'conv2d' in param_set['layer']:
            code_string = get_c_code_conv('_conv' + str(conv_cnt), cmsis_nn_buffers_map[i], is_first_layer)
            file_string += code_string
            conv_cnt += 1
        elif 'batch_normalization' in param_set['layer']:
            code_string = get_c_code_dw_conv('_dw_cnv' + str(dw_conv_cnt), cmsis_nn_buffers_map[i], is_first_layer)
            file_string += code_string
            dw_conv_cnt += 1
        elif 'average_pooling2d' in param_set['layer']:
            code_string = get_c_code_avg_pool('_avg_pool' + str(avg_pool_cnt), cmsis_nn_buffers_map[i], is_first_layer)
            file_string += code_string
            avg_pool_cnt += 1
        elif 'dense' in param_set['layer']:
            code_string = get_c_code_fc('_fc' + str(fc_cnt), cmsis_nn_buffers_map[i], is_first_layer)
            file_string += code_string
            fc_cnt += 1
        elif 'Identity' in param_set['layer']:
            code_string = get_c_code_softmax('_softmax' + str(softmax_cnt), cmsis_nn_buffers_map[i], is_first_layer)
            file_string += code_string
            softmax_cnt = softmax_cnt + 1
        elif 'add' in param_set['layer']:
            code_string = get_c_code_add('_add' + str(add_cnt), cmsis_nn_buffers_map[i])
            file_string += code_string
            add_cnt += 1

    if model_id == 'ano':
        ann_output_buffer = cmsis_nn_buffers_map[i]['output'][0]
        file_string += (
            '\tfor(uint16_t i=0; i<INPUT_DATA_SIZE; i++){\n'
                '\t\tdequantize(&(buffer' + str(ann_output_buffer) + '[i]), ANN_OUT_QUANT_SCALE, ANN_OUT_QUANT_ZP, 1, &ann_out_f);\n'
                '\t\terror = ann_in_f[i] - ann_out_f;\n'
                '\t\terror = error * error;\n'
                '\t\tprediction += error;\n'
            '\t}\n\n'

            '\tprediction = prediction / INPUT_DATA_SIZE;\n\n'

            '\treturn prediction;\n'
            '}\n\n'
        )
    else:
        idx_buf_map = len(cmsis_nn_buffers_map) - 1
        buffer_output_idx = cmsis_nn_buffers_map[idx_buf_map]['output'][0]
        file_string += (
            '\n\tfor(uint16_t i=0; i<OUTPUT_DATA_SIZE; i++){\n'
                '\t\tif(buffer' + str(buffer_output_idx) + '[i] > buffer' + str(buffer_output_idx) + '[prediction]){\n'
                    '\t\t\tprediction = i;\n'
                '\t\t}\n'
            '\t}\n\n'
            '\treturn prediction;\n'
            '}\n\n'
        )

    return file_string

##################################################################################
############################## AUXILIAR FUNCTIONS ################################
def get_c_code_ctx_size_conv(sufix, var_name):
    string = (
        '\t' + var_name + ' = arm_convolve_wrapper_s8_get_buffer_size(&conv_params' + sufix + ', &input_dims' + sufix +
        ', &filter_dims' + sufix + ', &output_dims' + sufix + ');\n'
    )

    return string


def get_c_code_ctx_size_dw_conv(sufix, var_name):
    string = (
        '\t' + var_name + ' = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params' + sufix +
        ', &input_dims' + sufix + ', &filter_dims' + sufix + ', &output_dims' + sufix + ');\n'
    )

    return string


def get_c_code_ctx_size_avg_pool(sufix, var_name):
    string = (
        '\t' + var_name + ' = arm_avgpool_s8_get_buffer_size(output_dims' + sufix + '.w,' +
        'input_dims' + sufix + '.c);\n'
    )

    return string


def get_c_code_conv(sufix, cmsis_nn_buf_map, is_first_layer):
    string = get_c_code_ctx_size_conv(sufix, 'ctx.size')
    i1 = cmsis_nn_buf_map['input'][0]
    o1 = cmsis_nn_buf_map['output'][0]

    string += ('\tarm_convolve_wrapper_s8(&ctx, &conv_params' + sufix + ', &quant_params' + sufix + ', &input_dims' + sufix)
    if is_first_layer:
        string += (', input_data')
    else:
        string += (', buffer' + str(i1))
    
    string += (
        ', &filter_dims' + sufix + ', wt' + sufix + ' ,&bias_dims' + sufix +
        ', bias' + sufix + ', &output_dims' + sufix + ', buffer' + str(o1) + ');\n'
    )

    if len(cmsis_nn_buf_map['output']) == 2:
        o2 = cmsis_nn_buf_map['output'][1]
        string += (
            '\tmemcpy(buffer' + str(o2) + ', buffer' + str(o1) + ', output_size' + sufix + ');\n'
         )
    
    string += ('\n')

    return string


def get_c_code_dw_conv(sufix, cmsis_nn_buf_map, is_first_layer):
    string = get_c_code_ctx_size_dw_conv(sufix, 'ctx.size')
    i1 = cmsis_nn_buf_map['input'][0]
    o1 = cmsis_nn_buf_map['output'][0]

    string += ('\tarm_depthwise_conv_wrapper_s8(&ctx, &dw_conv_params' + sufix + ', &quant_params' + sufix + ', &input_dims' + sufix)

    if is_first_layer:
        string += (', input_data')
    else:
        string += (', buffer' + str(i1))
    
    string += (
        ', &filter_dims' + sufix + ', wt' + sufix + ' ,&bias_dims' + sufix +
        ', bias' + sufix + ', &output_dims' + sufix + ', buffer' + str(o1) + ');\n\n'
    )

    return string


def get_c_code_avg_pool(sufix, cmsis_nn_buf_map, is_first_layer):
    string = get_c_code_ctx_size_avg_pool(sufix, 'ctx.size')
    i1 = cmsis_nn_buf_map['input'][0]
    o1 = cmsis_nn_buf_map['output'][0]

    string += ('\tarm_avgpool_s8(&ctx, &pool_params' + sufix + ', &input_dims' + sufix)
     
    if is_first_layer:
        string += (', input_data')
    else:
        string += (', buffer' + str(i1))
    
    string += (', &filter_dims' + sufix + ', &output_dims' + sufix + ', buffer' + str(o1) + ');\n\n')

    return string


def get_c_code_fc(sufix, cmsis_nn_buf_map, is_first_layer):
    i1 = cmsis_nn_buf_map['input'][0]
    o1 = cmsis_nn_buf_map['output'][0]

    string = (
        '\tctx.size = 0;\n'
        '\tarm_fully_connected_s8(&ctx, &fc_params' + sufix + ', &quant_params' + sufix + ', &input_dims' + sufix
    )

    if is_first_layer:
        string += (', input_data')
    else:
        string += (', buffer' + str(i1))
    
    string += (
        ', &filter_dims' + sufix + ', wt' + sufix + ' ,&bias_dims' + sufix +
        ', bias' + sufix + ', &output_dims' + sufix + ', buffer' + str(o1) + ');\n\n'
    )

    return string


def get_c_code_softmax(sufix, cmsis_nn_buf_map, is_first_layer):
    i1 = cmsis_nn_buf_map['input'][0]
    o1 = cmsis_nn_buf_map['output'][0]

    if is_first_layer:
        string = ('\tarm_softmax_s8(input_data')
    else:
        string = ('\tarm_softmax_s8(buffer' + str(i1))

    string += (
        ', num_rows' + sufix + ', row_size' + sufix + ', mult' + sufix +
        ', shift' + sufix + ', diff_min' + sufix + ', buffer' + str(o1) + ');\n\n'
    )

    return string


def get_c_code_add(sufix, cmsis_nn_buf_map):
    i1 = cmsis_nn_buf_map['input'][0]
    i2 = cmsis_nn_buf_map['input'][1]
    o1 = cmsis_nn_buf_map['output'][0]

    string = (
        '\tarm_elementwise_add_s8(buffer' + str(i1) + ', buffer' + str (i2) + ', input_1_offset' + sufix + ', '
        'input_1_mult' + sufix + ', input_1_shift' + sufix + ', input_2_offset' + sufix + ', input_2_mult' + sufix + ', '
        'input_2_shift' + sufix + ', left_shift' + sufix + ', buffer' + str(o1) + ', out_offset' + sufix + ', '
        'out_mult' + sufix + ', out_shift' + sufix + ', out_activation_min' + sufix + ', out_activation_max' + sufix + ', '
        'block_size' + sufix + ');\n'
    )

    if len(cmsis_nn_buf_map['output']) == 2:
        o2 = cmsis_nn_buf_map['output'][1]
        string += (
            '\tmemcpy(buffer' + str(o2) + ', buffer' + str(o1) + ', block_size' + sufix + ');\n'
        ) 

    string += ('\n')

    return string
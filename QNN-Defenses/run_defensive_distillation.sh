#!/bin/bash
ANN_CIFAR_FLOAT='models/cifar_resnet_float.h5'
QNN_CIFAR_INT16='models/cifar_resnet_int16.tflite'
QNN_CIFAR_INT8='models/cifar_resnet_int8.tflite'

ANN_VWW_FLOAT='models/vww_float.h5'
QNN_VWW_INT16='models/vww_int16.tflite'
QNN_VWW_INT8='models/vww_int8.tflite'

ANN_COFFEE_FLOAT='models/coffee_cnn_float.h5'
QNN_COFFEE_INT16='models/coffee_cnn_int16.tflite'
QNN_COFFEE_INT8='models/coffee_cnn_int8.tflite'

python3 defensive_distillation.py --dataset_id cifar10 --batch_size 32 --epochs 200 --workers 8 --ann_source_float "$ANN_CIFAR_FLOAT" \
                    --qnn_source_int16 "$QNN_CIFAR_INT16" --qnn_source_int8 "$QNN_CIFAR_INT8" --ann_new_float new_models/cifar_distil_train.h5 \
                    --qnn_new_int16 new_models/cifar_distil_int16.tflite  --qnn_new_int8 new_models/cifar_distil_int8.tflite \
                    --softmax_temp 20 --alpha 0.1

python3 defensive_distillation.py --dataset_id vww --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_VWW_FLOAT" \
                    --qnn_source_int16 "$QNN_VWW_INT16" --qnn_source_int8 "$QNN_VWW_INT8" --ann_new_float new_models/vww_distil.h5 \
                    --qnn_new_int16 new_models/vww_distil_int16.tflite  --qnn_new_int8 new_models/vww_distil_int8.tflite \
                    --softmax_temp 20 --alpha 0.1

python3 defensive_distillation.py --dataset_id coffee --batch_size 32 --epochs 25 --workers 8 --ann_source_float "$ANN_COFFEE_FLOAT" \
                    --qnn_source_int16 "$QNN_COFFEE_INT16" --qnn_source_int8 "$QNN_COFFEE_INT8" --ann_new_float new_models/coffee_distil.h5 \
                    --qnn_new_int16 new_models/coffee_distil_int16.tflite  --qnn_new_int8 new_models/coffee_distil_int8.tflite \
                    --softmax_temp 20 --alpha 0.1
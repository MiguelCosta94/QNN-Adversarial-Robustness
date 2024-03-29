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

python3 adv_sat.py --dataset_id cifar10 --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_CIFAR_FLOAT" \
                    --qnn_source_int16 "$QNN_CIFAR_INT16" --qnn_source_int8 "$QNN_CIFAR_INT8" --ann_new_float new_models/cifar_sat.h5 \
                    --qnn_new_int16 new_models/cifar_sat_int16.tflite  --qnn_new_int8 new_models/cifar_sat_int8.tflite \
                    --sink_reduction sum --sink_epsilon 1.0 --sink_max_iter 50 --pgd_eps 0.008 --pgd_max_iter 10

python3 adv_sat.py --dataset_id vww --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_VWW_FLOAT" \
                    --qnn_source_int16 "$QNN_VWW_INT16" --qnn_source_int8 "$QNN_VWW_INT8" --ann_new_float new_models/vww_adv_sat.h5 \
                    --qnn_new_int16 new_models/vww_sat_int16.tflite  --qnn_new_int8 new_models/vww_sat_int8.tflite \
                    --sink_reduction sum --sink_epsilon 1.0 --sink_max_iter 50 --pgd_eps 0.008 --pgd_max_iter 10

python3 adv_sat.py --dataset_id coffee --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_COFFEE_FLOAT" \
                    --qnn_source_int16 "$QNN_COFFEE_INT16" --qnn_source_int8 "$QNN_COFFEE_INT8" --ann_new_float new_models/coffee_sat.h5 \
                    --qnn_new_int16 new_models/coffee_sat_int16.tflite  --qnn_new_int8 new_models/coffee_sat_int8.tflite \
                    --sink_reduction sum --sink_epsilon 1.0 --sink_max_iter 50 --pgd_eps 0.008 --pgd_max_iter 10
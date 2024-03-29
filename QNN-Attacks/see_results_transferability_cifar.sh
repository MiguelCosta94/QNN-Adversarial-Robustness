#!/bin/bash
DATASET_ID=cifar10
NUM_CLASSES=10
ANN_FLOAT='models/cifar_resnet_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_adv_train_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_distil_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_ens_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_sat_float.h5'

QNN_INT16='models/cifar_resnet_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_adv_train_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_distil_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_ens_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_sat_int16.tflite'

QNN_INT8='models/cifar_resnet_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_adv_train_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_sat_int8.tflite'

echo "=============================================================="
echo "======================PGD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/pgd'

echo "=============================================================="
echo "======================DEEPFOOL================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/deepfool'

echo "=============================================================="
echo "======================JSMA===================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/jsma'

echo "=============================================================="
echo "======================CW-L2==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/cwl2'

echo "=============================================================="
echo "======================CW-LINF================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/cwlinf'

echo "=============================================================="
echo "======================EAD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/ead'

echo "=============================================================="
echo "======================AUTOATTACK=============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/autoattack'

echo "=============================================================="
echo "======================ZOO====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/zoo'

echo "=============================================================="
echo "======================SQUARE-L2==============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/square_l2'

echo "=============================================================="
echo "======================SQUARE-LINF============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/square_linf'


echo "=============================================================="
echo "======================BOUNDARY================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/boundary'

echo "=============================================================="
echo "======================GEODA==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/cifar/geoda'
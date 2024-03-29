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
echo "======================ZOO-FLOAT==============================="
python3 see_results_attack_float.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --logs_dir 'logs/cifar/zoo'

echo "=============================================================="
echo "======================ZOO-INT16================================"
python3 see_results_attack_int16.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16"  \
                    --logs_dir 'logs/cifar/zoo_int16'

echo "=============================================================="
echo "======================ZOO-INT8================================"
python3 see_results_attack_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8"  \
                    --logs_dir 'logs/cifar/zoo_int8'

echo "=============================================================="
echo "======================SQUARE-L2-FLOAT========================="
python3 see_results_attack_float.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --logs_dir 'logs/cifar/square_l2'

echo "=============================================================="
echo "======================SQUARE-L2-INT16========================="
python3 see_results_attack_int16.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16"  \
                    --logs_dir 'logs/cifar/square_l2_int16'

echo "=============================================================="
echo "======================SQUARE-L2-INT8=========================="
python3 see_results_attack_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8"  \
                    --logs_dir 'logs/cifar/square_l2_int8'

echo "=============================================================="
echo "======================SQUARE-LINF-FLOAT======================="
python3 see_results_attack_float.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --logs_dir 'logs/cifar/square_linf'

echo "=============================================================="
echo "======================SQUARE-LINF-INT16======================="
python3 see_results_attack_int16.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16"  \
                    --logs_dir 'logs/cifar/square_linf_int16'

echo "=============================================================="
echo "======================SQUARE-LINF-INT8========================"
python3 see_results_attack_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8"  \
                    --logs_dir 'logs/cifar/square_linf_int8'

echo "=============================================================="
echo "======================BOUNDARY-FLOAT=========================="
python3 see_results_attack_float.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --logs_dir 'logs/cifar/boundary'

echo "=============================================================="
echo "======================BOUNDARY-INT16=========================="
python3 see_results_attack_int16.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16"  \
                    --logs_dir 'logs/cifar/boundary_int16'

echo "=============================================================="
echo "======================BOUNDARY-INT8==========================="
python3 see_results_attack_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8"  \
                    --logs_dir 'logs/cifar/boundary_int8'

echo "=============================================================="
echo "======================GEODA-FLOAT32==========================="
python3 see_results_attack_float.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --logs_dir 'logs/cifar/geoda'

echo "=============================================================="
echo "======================GEODA-INT16============================="
python3 see_results_attack_int16.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16"  \
                    --logs_dir 'logs/cifar/geoda_int16'

echo "=============================================================="
echo "======================GEODA-INT8=============================="
python3 see_results_attack_int8.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8"  \
                    --logs_dir 'logs/cifar/geoda_int8'
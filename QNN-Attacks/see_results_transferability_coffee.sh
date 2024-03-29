#!/bin/bash
DATASET_ID=coffee
NUM_CLASSES=4

ANN_FLOAT='models/coffee_cnn_float.h5'
#ANN_FLOAT='models-defense_enhanced/coffee_adv_train_float.h5'
#ANN_FLOAT='models-defense_enhanced/coffee_distil_float.h5'
#ANN_FLOAT='models-defense_enhanced/coffee_ens_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/coffee_sat_float.h5'

QNN_INT16='models/coffee_cnn_int16.tflite'
#QNN_INT16='models-defense_enhanced/coffee_adv_train_int16.tflite'
#QNN_INT16='models-defense_enhanced/coffee_distil_int16.tflite'
#QNN_INT16='models-defense_enhanced/coffee_ens_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/coffee_sat_int16.tflite'

QNN_INT8='models/coffee_cnn_int8.tflite'
#QNN_INT8='models-defense_enhanced/coffee_adv_train_int8.tflite'
#QNN_INT8='models-defense_enhanced/coffee_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/coffee_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/coffee_sat_int8.tflite'

echo "=============================================================="
echo "======================PGD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/pgd'

echo "=============================================================="
echo "======================DEEPFOOL================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/deepfool'

echo "=============================================================="
echo "======================JSMA===================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/jsma'

echo "=============================================================="
echo "======================CW-L2==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/cwl2'

echo "=============================================================="
echo "======================CW-LINF================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/cwlinf'

echo "=============================================================="
echo "======================EAD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/ead'

echo "=============================================================="
echo "======================AUTOATTACK=============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/autoattack'

echo "=============================================================="
echo "======================ZOO====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/zoo'

echo "=============================================================="
echo "======================SQUARE-L2==============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/square_l2'

echo "=============================================================="
echo "======================SQUARE-LINF============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/square_linf'


echo "=============================================================="
echo "======================BOUNDARY================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/boundary'

echo "=============================================================="
echo "======================GEODA==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/sink_adv_train/coffee/geoda'
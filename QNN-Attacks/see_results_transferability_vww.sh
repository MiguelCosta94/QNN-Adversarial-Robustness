#!/bin/bash
DATASET_ID=vww
NUM_CLASSES=2
ANN_FLOAT='models/vww_float.h5'
#ANN_FLOAT='models-defense_enhanced/vww_adv_train_float.h5'
#ANN_FLOAT='models-defense_enhanced/vww_distil_float.h5'
#ANN_FLOAT='models-defense_enhanced/vww_ens_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/vww_sat_float.h5'

QNN_INT16='models/vww_int16.tflite'
#QNN_INT16='models-defense_enhanced/vww_adv_train_int16.tflite'
#QNN_INT16='models-defense_enhanced/vww_distil_int16.tflite'
#QNN_INT16='models-defense_enhanced/vww_ens_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/vww_sat_int16.tflite'

QNN_INT8='models/vww_int8.tflite'
#QNN_INT8='models-defense_enhanced/vww_adv_train_int8.tflite'
#QNN_INT8='models-defense_enhanced/vww_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/vww_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/vww_sat_int8.tflite'

echo "=============================================================="
echo "======================PGD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/pgd'

echo "=============================================================="
echo "======================DEEPFOOL================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/deepfool'

echo "=============================================================="
echo "======================JSMA===================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/jsma'

echo "=============================================================="
echo "======================CW-L2==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/cwl2'

echo "=============================================================="
echo "======================CW-LINF================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/cwlinf'

echo "=============================================================="
echo "======================EAD====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/ead'

echo "=============================================================="
echo "======================AUTOATTACK=============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/autoattack'

echo "=============================================================="
echo "======================ZOO====================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/zoo'

echo "=============================================================="
echo "======================SQUARE-L2==============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/square_l2'

echo "=============================================================="
echo "======================SQUARE-LINF============================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/square_linf'


echo "=============================================================="
echo "======================BOUNDARY================================"
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/boundary'

echo "=============================================================="
echo "======================GEODA==================================="
python3 see_results_transferability.py --dataset_id "$DATASET_ID" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_dir 'logs/vww/geoda'




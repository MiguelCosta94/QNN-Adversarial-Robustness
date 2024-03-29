#!/bin/bash
DATASET_ID=cifar10
DATASET_SIZE=1000
NUM_CLASSES=10
ANN_FLOAT='models/cifar_resnet_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_distil_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_ens_adv_float.h5'
#ANN_FLOAT='models-defense_enhanced/cifar_sat_float.h5'
QNN_INT16='models/cifar_resnet_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_distil_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_ens_adv_int16.tflite'
#QNN_INT16='models-defense_enhanced/cifar_sat_int16.tflite'
QNN_INT8='models/cifar_resnet_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_distil_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_ens_adv_int8.tflite'
#QNN_INT8='models-defense_enhanced/cifar_sat_int8.tflite'

echo "=============================================================="
echo "======================AUTOATTACK=============================="
python3 autoattack.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT" \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/autoattack' --autoattack_eps 0.004

echo "=============================================================="
echo "======================BOUNDARY================================"
python3 boundary.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                    --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/boundary' --boundary_delta 0.1 \
                    --boundary_epsilon 1.0 --boundary_max_iter 500

echo "=============================================================="
echo "======================BOUNDARY-INT8==========================="
python3 boundary_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                        --logs_folder 'logs/cifar/boundary_int8' --boundary_delta 0.1 --boundary_epsilon 1.0 --boundary_max_iter 500

echo "=============================================================="
echo "======================BOUNDARY-INT16=========================="
python3 boundary_int16.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16" \
                       --logs_folder 'logs/cifar/boundary_int16' --boundary_delta 0.1 --boundary_epsilon 1.0 --boundary_max_iter 500

echo "=============================================================="
echo "======================CW-L2==================================="
python3 cwl2.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/cwl2' --cwl2_max_iter 10 \
                --cwl2_binary_search_steps 10 --cwl2_initial_const 0.01

echo "=============================================================="
echo "======================CW-LINF================================="
python3 cwlinf.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/cwlinf' --cwlinf_max_iter 10 --cwlinf_learning_rate 0.01

echo "=============================================================="
echo "======================DEEPFOOL================================"
python3 deepfool.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/deepfool' --deepfool_max_iter 100\
                --deepfool_epsilon 0.008

echo "=============================================================="
echo "======================EAD====================================="
python3 ead.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/ead' --ead_max_iter 10 \
                --ead_binary_search_steps 10 --ead_initial_const 0.01

echo "=============================================================="
echo "======================GEODA==================================="
python3 geoda.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/geoda' --geoda_bin_search_tol 0.0001 \
                --geoda_sub_dim 75

echo "=============================================================="
echo "======================GEODA-INT8=============================="
python3 geoda_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                    --logs_folder 'logs/cifar/geoda_int8' --geoda_bin_search_tol 0.0001 --geoda_sub_dim 75

echo "=============================================================="
echo "======================GEODA-INT16============================="
python3 geoda_int16.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16" \
                    --logs_folder 'logs/cifar/geoda_int16' --geoda_bin_search_tol 0.0001 --geoda_sub_dim 75

echo "=============================================================="
echo "======================JSMA===================================="
python3 jsma.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/jsma' --jsma_gamma 1.0 \
                --jsma_theta "0.08"

echo "=============================================================="
echo "======================PGD====================================="
python3 pgd.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/pgd' --pgd_num_random_init 20 --pgd_max_iter 20 \
                --pgd_eps "0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008"

echo "=============================================================="
echo "======================SQUARE-L2==============================="
python3 square.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/square_l2' --square_norm 2 --square_max_iter 2000 \
                --square_eps 1.0 --square_p_init 0.8

echo "=============================================================="
echo "======================SQUARE-LINF============================="
python3 square.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/square_linf' --square_norm "inf" --square_max_iter 1000 \
                --square_eps 0.015 --square_p_init 0.05

echo "=============================================================="
echo "======================SQUARE-L2-INT8=========================="
python3 square_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                --logs_folder 'logs/cifar/square_l2_int8' --square_norm 2 --square_max_iter 2000 --square_eps 1.0 --square_p_init 0.8

echo "=============================================================="
echo "======================SQUARE-LINF-INT8========================"
python3 square_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                --logs_folder 'logs/cifar/square_linf_int8' --square_norm "inf" --square_max_iter 1000 --square_eps 0.015 --square_p_init 0.05

echo "=============================================================="
echo "======================SQUARE-L2-INT16========================="
python3 square_int16.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16" \
                --logs_folder 'logs/cifar/square_l2_int16' --square_norm 2 --square_max_iter 2000 --square_eps 1.0 --square_p_init 0.8

echo "=============================================================="
echo "======================SQUARE-LINF-INT16======================="
python3 square_int16.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16" \
                --logs_folder 'logs/cifar/square_linf_int16' --square_norm "inf" --square_max_iter 1000 --square_eps 0.015 --square_p_init 0.05

echo "=============================================================="
echo "======================ZOO====================================="
python3 zoo.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --ann_float "$ANN_FLOAT"  \
                --qnn_int16 "$QNN_INT16" --qnn_int8 "$QNN_INT8" --logs_folder 'logs/cifar/zoo' --zoo_max_iter 10 --zoo_binary_search_steps 5 \
                --zoo_initial_const 0.01

echo "=============================================================="
echo "======================ZOO-INT8================================"
python3 zoo_int8.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int8 "$QNN_INT8" \
                    --logs_folder 'logs/cifar/zoo_int8' --zoo_max_iter 10 --zoo_binary_search_steps 5 --zoo_initial_const 0.01

echo "=============================================================="
echo "======================ZOO-INT16==============================="
python3 zoo_int16.py --dataset_id "$DATASET_ID" --dataset_size "$DATASET_SIZE" --num_classes "$NUM_CLASSES" --qnn_int16 "$QNN_INT16" \
                    --logs_folder 'logs/cifar/zoo_int16' --zoo_max_iter 10 --zoo_binary_search_steps 5 --zoo_initial_const 0.01
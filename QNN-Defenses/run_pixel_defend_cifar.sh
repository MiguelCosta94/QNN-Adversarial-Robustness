#!/bin/bash
ANN_FLOAT='models/cifar_resnet_float.h5'
QNN_INT16='models/cifar_resnet_int16.tflite'
QNN_INT8='models/cifar_resnet_int8.tflite'

PIXEL_CNN='pixel_cnn/px_cnn/pixel_cnn_cifar'
PD_EPS=38

echo "=======================DEEPFOOL==============================="
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=======================CW-LINF================================"
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=======================AUTOATTACK================================"
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=======================SQUARE_L2=============================="
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/square_l2/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/square_l2_int16/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/square_l2_int8/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=======================SQUARE_LINF============================"
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/square_linf/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/square_linf_int16/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/square_linf_int8/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=======================BOUNDARY==============================="
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/boundary/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/boundary_int16/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/boundary_int8/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"


echo "=========================GEODA================================"
echo "=========================FLOAT================================"
python3 pixel_defend_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/geoda/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-16==============================="
python3 pixel_defend_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/geoda_int16/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"

echo "=========================INT-8================================"
python3 pixel_defend_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/geoda_int8/" \
                        --pixel_cnn "$PIXEL_CNN" --pd_eps "$PD_EPS"
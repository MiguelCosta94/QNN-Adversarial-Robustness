#!/bin/bash
ANN_FLOAT='models/cifar_resnet_float.h5'
QNN_INT16='models/cifar_resnet_int16.tflite'
QNN_INT8='models/cifar_resnet_int8.tflite'

FT_THRESHOLD_FLOAT=1.7634952
FT_THRESHOLD_INT16=48802.0
FT_THRESHOLD_INT8=260.0
FT_COLOR_BIT_FLOAT=4
FT_COLOR_BIT_INT16=16
FT_COLOR_BIT_INT8=8
FT_FPR=0.05
FT_SMOOTH_WINDOW_SIZE=2


echo "=======================DEEPFOOL==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================CW-LINF==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================AUTOATTACK============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================SQUARE-L2=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/square_l2/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/cifar/square_l2_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/cifar/square_l2_int8/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "======================SQUARE-LINF============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/square_linf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/cifar/square_linf_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/cifar/square_linf_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================BOUNDARY=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/boundary/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/cifar/boundary_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/cifar/boundary_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================GEODA=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id cifar10 --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/cifar/geoda/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id cifar10 --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/cifar/geoda_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id cifar10 --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/cifar/geoda_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"
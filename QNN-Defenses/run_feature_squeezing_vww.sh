#!/bin/bash
ANN_FLOAT='models/vww_float.h5'
QNN_INT16='models/vww_int16.tflite'
QNN_INT8='models/vww_int8.tflite'

FT_THRESHOLD_FLOAT=1.5573368
FT_THRESHOLD_INT16=50179.0
FT_THRESHOLD_INT8=242.0
FT_COLOR_BIT_FLOAT=8
FT_COLOR_BIT_INT16=16
FT_COLOR_BIT_INT8=8
FT_FPR=0.05
FT_SMOOTH_WINDOW_SIZE=6


echo "=======================DEEPFOOL==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id vww --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/vww/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id vww --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/vww/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id vww --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/vww/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================CW-LINF==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id vww --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/vww/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id vww --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/vww/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id vww --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/vww/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================AUTOATTACK============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id vww --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/vww/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id vww --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/vww/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id vww --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/vww/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================SQUARE-L2=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id vww --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/vww/square_l2/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id vww --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/vww/square_l2_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id vww --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/vww/square_l2_int8/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "======================SQUARE-LINF============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id vww --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/vww/square_linf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id vww --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/vww/square_linf_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id vww --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/vww/square_linf_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"
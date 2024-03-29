#!/bin/bash
ANN_FLOAT='models/coffee_cnn_float.h5'
QNN_INT16='models/coffee_cnn_int16.tflite'
QNN_INT8='models/coffee_cnn_int8.tflite'

FT_THRESHOLD_FLOAT=0.11576004
FT_THRESHOLD_INT16=3619.0
FT_THRESHOLD_INT8=34.0
FT_COLOR_BIT_FLOAT=8
FT_COLOR_BIT_INT16=16
FT_COLOR_BIT_INT8=8
FT_FPR=0.05
FT_SMOOTH_WINDOW_SIZE=2


echo "=======================DEEPFOOL==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/coffee/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/coffee/deepfool/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================CW-LINF==============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/coffee/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/coffee/cwlinf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================AUTOATTACK============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path "../QNN-Attacks/logs/coffee/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/coffee/autoattack/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "=======================SQUARE-L2=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/square_l2/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/coffee/square_l2_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path "../QNN-Attacks/logs/coffee/square_l2_int8/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "======================SQUARE-LINF============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/square_linf/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/coffee/square_linf_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/coffee/square_linf_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================BOUNDARY=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/boundary/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/coffee/boundary_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/coffee/boundary_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" \
                        --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"


echo "========================GEODA=============================="
echo "=========================FLOAT================================"
python3 feature_squeezing_float.py --dataset_id coffee --ann_source_float "$ANN_FLOAT" --datasets_path "../QNN-Attacks/logs/coffee/geoda/" \
                        --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_FLOAT" --ft_fpr "$FT_FPR" --ft_color_bit "$FT_COLOR_BIT_FLOAT" \
                        --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-16==============================="
python3 feature_squeezing_int16.py --dataset_id coffee --qnn_source_int16 "$QNN_INT16" --datasets_path \
                        "../QNN-Attacks/logs/coffee/geoda_int16/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT16" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT16" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"

echo "=========================INT-8================================"
python3 feature_squeezing_int8.py --dataset_id coffee --qnn_source_int8 "$QNN_INT8" --datasets_path \
                        "../QNN-Attacks/logs/coffee/geoda_int8/" --ft_calculate_threshold False --ft_threshold "$FT_THRESHOLD_INT8" --ft_fpr "$FT_FPR" \
                        --ft_color_bit "$FT_COLOR_BIT_INT8" --ft_smooth_window_size "$FT_SMOOTH_WINDOW_SIZE"
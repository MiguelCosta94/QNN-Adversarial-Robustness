#!/bin/bash
ANN_CIFAR_FLOAT='models/cifar_resnet_float.h5'
QNN_CIFAR_INT16='models/cifar_resnet_int16.tflite'
QNN_CIFAR_INT8='models/cifar_resnet_int8.tflite'
ANN_CIFAR_ATTACK_1='models_ensemble_adv_train/cifar_mobilenet_large.h5'
ANN_CIFAR_ATTACK_2='models_ensemble_adv_train/cifar_vgg16.h5'

ANN_VWW_FLOAT='models/vww_float.h5'
QNN_VWW_INT16='models/vww_int16.tflite'
QNN_VWW_INT8='models/vww_int8.tflite'
ANN_VWW_ATTACK_1='models_ensemble_adv_train/vww_cnn.h5'
ANN_VWW_ATTACK_2='models_ensemble_adv_train/vww_resnet_eembc.h5'

ANN_COFFEE_FLOAT='models/coffee_cnn_float.h5'
QNN_COFFEE_INT16='models/coffee_cnn_int16.tflite'
QNN_COFFEE_INT8='models/coffee_cnn_int8.tflite'
ANN_COFFEE_ATTACK_1='models_ensemble_adv_train/coffee_mobilenet_v1.h5'
ANN_COFFEE_ATTACK_2='models_ensemble_adv_train/coffee_resnet_eembc.h5'

python3 ensemble_adversarial_training.py --dataset_id cifar10 --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_CIFAR_FLOAT" \
                    --qnn_source_int16 "$QNN_CIFAR_INT16" --qnn_source_int8 "$QNN_CIFAR_INT8" --ann_new_float new_models/cifar_ens_adv_train.h5 \
                    --qnn_new_int16 new_models/cifar_ens_adv_train_int16.tflite  --qnn_new_int8 new_models/cifar_ens_adv_train_int8.tflite \
                    --ann_attack_1 "$ANN_CIFAR_ATTACK_1" --ann_attack_2 "$ANN_CIFAR_ATTACK_2" --fgsm_eps 0.008 \
                    --pgd_eps 0.008 --pgd_num_random_init 1 --pgd_max_iter 10

python3 ensemble_adversarial_training.py --dataset_id vww --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_VWW_FLOAT" \
                    --qnn_source_int16 "$QNN_VWW_INT16" --qnn_source_int8 "$QNN_VWW_INT8" --ann_new_float new_models/vww_ens_adv_train.h5 \
                    --qnn_new_int16 new_models/vww_ens_adv_train_int16.tflite  --qnn_new_int8 new_models/vww_ens_adv_train_int8.tflite \
                    --ann_attack_1 "$ANN_VWW_ATTACK_1" --ann_attack_2 "$ANN_VWW_ATTACK_2" --fgsm_eps 0.008 \
                    --pgd_eps 0.008 --pgd_num_random_init 1 --pgd_max_iter 10

python3 ensemble_adversarial_training.py --dataset_id coffee --batch_size 32 --epochs 50 --workers 8 --ann_source_float "$ANN_COFFEE_FLOAT" \
                    --qnn_source_int16 "$QNN_COFFEE_INT16" --qnn_source_int8 "$QNN_COFFEE_INT8" --ann_new_float new_models/coffee_ens_adv_train.h5 \
                    --qnn_new_int16 new_models/coffee_ens_adv_train_int16.tflite  --qnn_new_int8 new_models/coffee_ens_adv_train_int8.tflite \
                    --ann_attack_1 "$ANN_COFFEE_ATTACK_1" --ann_attack_2 "$ANN_COFFEE_ATTACK_2" --fgsm_eps 0.008 \
                    --pgd_eps 0.008 --pgd_num_random_init 1 --pgd_max_iter 10
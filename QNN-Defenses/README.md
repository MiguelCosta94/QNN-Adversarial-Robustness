# ANN-Defenses
This repository allows the reproducibility of the results associated with the paper "**David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**".


## Directory Structure

| Directory | Content |
| ------ | ------ |
| /art | ART framework including attacks/defenses refactored for int-16 and int-8 data |
| /defenses | Contains defenses not implemented (or fully implemented) in ART framework |
| /models | Set of ANNs and QNNs to be attacked |
| /models_ensemble_adv_train | Set of auxiliary models for which the ensemble adversarial training defense generates adversarial examples |
| /pixel_cnn | Pixel-CNN used in Pixel Defend |
| /tflite_to_cmsis | Framework to generate .c and .h files enabling the model execution on STM32 platforms using CMSIS-NN API |
| /utils | Backend functionality and dataset loading |
| run_adversarial_train.sh | Performs adversarial training on the CIFAR-10/VWW/COFFEE ANN and generates the respective QNNs |
| run_defensive_distillation.sh | Retrains the CIFAR-10/VWW/COFFEE ANN with defensive distillation and generates the respective QNNs |
| run_ensemble_adversarial_training.sh | Performs ensemble adversarial training on the CIFAR-10/VWW/COFFEE ANN and generates the respective QNNs |
| run_sat.sh | Performs Sinkhorn adversarial training on the CIFAR-10/VWW/COFFEE ANN and generates the respective QNNs |
| run_feature_squeezing_cifar.sh | Evaluates the adversarial robustness of ANNs/QNNs enhanced with Feature Squeezing before the input layer |
| run_pixel_defend_cifar.sh | Evaluates the adversarial robustness of ANNs/QNNs enhanced with Pixel Defend before the input layer |


## Requirements
- Python 3.9.18
- TensorFlow 2.13.0
- Remaining requirements are listed in requirements.txt


## Help and Support
### Communication
- E-mail: miguel.costa@dei.uminho.pt

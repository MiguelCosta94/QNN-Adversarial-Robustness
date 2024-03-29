# ANN-Attacks
This repository allows the reproducibility of the results associated with the paper "**David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**".


## Directory Structure

| Directory | Content |
| ------ | ------ |
| /art | ART framework including attacks/defenses refactored for int-16 and int-8 data |
| /autoattack | Implementation of AutoAttack in TF2 |
| /models | Set of ANNs/QNNs to be attacked |
| /models-defense_enhanced | Set of ANNs/QNNs enhanced with train-based defenses |
| /tflite_to_cmsis | Framework to generate .c and .h files enabling the model execution on STM32 platforms using CMSIS-NN API |
| /utils | Backend functionality and dataset loading |
| zero_density_estimated_grad_$dataset.py | Calculates the average zero density of the gradients estimated for ANNs and QNNs trained on $dataset |
| run_cifar_attacks.sh | Runs the full set of attacks against the CIFAR-10 ANNs/QNNs |
| see_results_direct_attacks_cifar.sh | Compares the success of direct attacks against floating-point ANNs with the success of direct attacks against QNNs |
| see_results_tranferability_cifar.sh | Evaluates how adversarial examples crafted on floating-point ANNs transfer to QNNs (accuracy and distortion) |


## Requirements
- Python 3.9.18
- TensorFlow 2.13.0
- Remaining requirements are listed in requirements.txt


## Help and Support
### Communication
- E-mail: miguel.costa@dei.uminho.pt

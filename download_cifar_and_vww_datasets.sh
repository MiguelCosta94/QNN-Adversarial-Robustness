#!/bin/bash
mkdir Datasets
cd Datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz
wget https://mega.nz/file/ufg2jISB#CA1ssv9fUozs3_dsfDz4PI8fNE0K-_eXYb9Uf79ShIU
tar -xvf new_coffee_dataset.tar.xz
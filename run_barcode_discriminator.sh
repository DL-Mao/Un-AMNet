#!/bin/bash

datapath="/f/SimpleNet-main/data/Code10"
datasets=('ACode' 'Code' 'EAN' 'QR' 'UPCA')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

echo "Starting barcode adaptive discriminator training..."
echo "Dataset path: $datapath"
echo "Training datasets: ${datasets[*]}"

python main.py \
--gpu 0 \
--seed 0 \
--log_group amnet_code10_test5_vision \
--log_project code10_melting_experiment \
--results_path results \
--run_name barcode_discriminator_run \
--save_segmentation_images \
net \
-b resnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 20 \
--embedding_size 256 \
--gan_epochs 2 \
--noise_std 0.02 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
--use_barcode_discriminator \
dataset \
--batch_size 8 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec "$datapath"

echo "Barcode adaptive discriminator training completed!"
echo "Results saved in: results/code10_BarcodeDiscriminator/amnet_code10_barcode/barcode_discriminator_run/"

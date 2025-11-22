# AMNet - Image Anomaly Detection and Localization

AMNet is a simple network for image anomaly detection and localization, based on the SimpleNet architecture.

## Environment Setup

This project requires Python 3.8+ and PyTorch. The environment used is PyTorch (Pim).

### Dependencies

- Python 3.8+
- PyTorch
- torchvision
- numpy
- click
- tqdm
- scikit-learn
- tensorboard

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SimpleNet-main2
```

2. Install required packages:
```bash
pip install torch torchvision numpy click tqdm scikit-learn tensorboard
```

## Dataset Preparation

The project supports two datasets:
- **MVTec AD**: Standard anomaly detection dataset
- **Code10**: Barcode anomaly detection dataset

Organize your dataset in the following structure:
```
data/
  └── Code10/
      ├── ACode/
      ├── Code/
      ├── EAN/
      ├── QR/
      └── UPCA/
```

## Training

### Basic Training

Train with standard discriminator:

```bash
python main.py \
  --gpu 0 \
  --seed 0 \
  --log_group amnet_code10 \
  --log_project code10_Results1 \
  --results_path results \
  --run_name run \
  --save_segmentation_images \
  net \
    -b resnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 50 \
    --embedding_size 256 \
    --gan_epochs 1 \
    --noise_std 0.015 \
    --dsc_hidden 1024 \
    --dsc_layers 2 \
    --dsc_margin 0.5 \
    --pre_proj 1 \
  dataset \
    --batch_size 8 \
    --resize 329 \
    --imagesize 288 \
    -d ACode -d Code -d EAN -d QR -d UPCA \
    mvtec /path/to/your/data
```

### Training with Barcode Adaptive Discriminator

Train with barcode-specific adaptive discriminator:

```bash
python main.py \
  --gpu 0 \
  --seed 0 \
  --log_group amnet_code10_barcode \
  --log_project code10_experiment \
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
    --dsc_margin 0.5 \
    --pre_proj 1 \
    --use_barcode_discriminator \
  dataset \
    --batch_size 8 \
    --resize 329 \
    --imagesize 288 \
    -d ACode -d Code -d EAN -d QR -d UPCA \
    mvtec /path/to/your/data
```

### Using Shell Scripts

Alternatively, you can use the provided shell scripts:

**Standard training:**
```bash
bash run2_1.sh
```

**Barcode discriminator training:**
```bash
bash run_barcode_discriminator.sh
```

Note: Remember to update the `datapath` variable in the shell scripts to point to your dataset location.

## Main Parameters

### General Options
- `--gpu`: GPU device ID (default: 0)
- `--seed`: Random seed (default: 0)
- `--results_path`: Path to save results (default: results)
- `--log_group`: Log group name
- `--log_project`: Log project name
- `--run_name`: Run name
- `--save_segmentation_images`: Save segmentation visualization images

### Network Options
- `-b, --backbone_names`: Backbone network (e.g., resnet50)
- `-le, --layers_to_extract_from`: Layers to extract features from (e.g., layer2, layer3)
- `--pretrain_embed_dimension`: Pretrained embedding dimension (default: 1536)
- `--target_embed_dimension`: Target embedding dimension (default: 1536)
- `--patchsize`: Patch size (default: 3)
- `--meta_epochs`: Number of meta epochs (default: 50)
- `--gan_epochs`: Number of GAN training epochs (default: 1)
- `--noise_std`: Noise standard deviation (default: 0.015)
- `--dsc_hidden`: Discriminator hidden dimension (default: 1024)
- `--dsc_layers`: Number of discriminator layers (default: 2)
- `--dsc_margin`: Discriminator margin (default: 0.5)
- `--pre_proj`: Use pre-projection (default: 0, set to 1 to enable)
- `--use_barcode_discriminator`: Use barcode adaptive discriminator

### Dataset Options
- `-d, --subdatasets`: Subdataset names (can specify multiple)
- `--batch_size`: Batch size (default: 8)
- `--resize`: Resize size (default: 329)
- `--imagesize`: Image size (default: 288)

## Results

Training results will be saved in the `results/` directory with the following structure:
```
results/
  └── <log_project>/
      └── <log_group>/
          └── <run_name>/
              ├── models/
              ├── segmentation_images/
              └── results.csv
```

## Citation

If you use this code, please cite the original SimpleNet paper:
```
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yixuan and Wang, Zilei},
  booktitle={CVPR},
  year={2023}
}
```


# AMNet - Image Anomaly Detection and Localization

AMNet is a simple network for image anomaly detection and localization, based on the SimpleNet architecture.

## Environment Setup

This project requires Python 3.8+ and PyTorch. The environment used is PyTorch.

### Dependencies

- Python 3.8+
- Torch = 2.4.1+cu118
- torchaudio = 2.4.1+cu118
- torchvision = 0.19.1+cu118
- numpy = 1.24.3
- click = 8.1.7
- tqdm = 4.67.0
- timm = 0.5.4
- scikit-learn = 1.0.2
- tensorboard = 2.14.0

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DL-Mao/Un-AMNet.git
cd Un-AMNet
```

2. Install required packages:

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.3 click==8.1.7 tqdm==4.67.0 timm==0.5.4 scikit-learn==1.0.2 tensorboard==2.14.0
```

## Dataset Preparation

The project supports two datasets:

- **MVTec AD**: [Standard anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Code10**: [Barcode anomaly detection dataset](https://www.kaggle.com/datasets/amor000/a-dataset-of-label-printing-defects)

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

If you use this code, please cite the original paper: *Adaptive mechanism-based unsupervised network for anomaly detection in printed labels.*

## Acknowledgement

Thank for great inspiration from [Simplenet](https://github.com/DonaldRR/SimpleNet).

# AMNet - Image Anomaly Detection and Localization

AMNet is a simple network for image anomaly detection and localization, based on the SimpleNet architecture.

## Environment Setup

This project requires Python 3.8+ and PyTorch. The environment used is PyTorch (Pim).

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

## Dataset Preparation

The project supports two datasets:
- **MVTec AD**: Standard anomaly detection dataset.(Link at: https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Code10**: Barcode anomaly detection dataset. (Link at: (https://www.kaggle.com/datasets/amor000/a-dataset-of-label-printing-defects))

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



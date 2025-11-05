# Image Extractor - Situation Recognition Model

This project implements a deep learning model for situation recognition using the ImSitu dataset and ACE2005 dataset. The model is designed to recognize situations in images by identifying verbs and their associated semantic roles.

## Project Structure

```
├── data/
│   ├── ace2005/          # ACE2005 dataset files
│   └── imsitu/           # ImSitu dataset
│       ├── images_256/   # Image data
│       ├── dev_annotations/
│       ├── train_annotations/
│       └── test_annotations/
├── models.py             # Neural network model architecture
├── data_loader.py        # Dataset loading and preprocessing
├── train.py             # Training script
└── preprocess_annotations.py  # Data preprocessing utilities
```

## Model Architecture

The project uses a `SituationRecognizer` model that consists of:
- ResNet50 backbone (pretrained on ImageNet) for feature extraction
- Verb prediction head
- Role embedding predictor for semantic role labeling
- Noun embeddings for argument prediction

## Requirements

- PyTorch
- torchvision
- PIL (Python Imaging Library)
- JSON

## Usage

1. **Data Preparation**:
   - Place the ImSitu dataset in the `data/imsitu` directory
   - Place the ACE2005 dataset in the `data/ace2005` directory

2. **Training**:
```bash
python train.py
```

3. **Dataset Loading**:
```python
from data_loader import ImSituDataset

# Initialize dataset
train_dataset = ImSituDataset(data_dir='./data/imsitu', split='train')
```

## Model Features

- Pretrained ResNet50 backbone
- Memory-efficient role prediction
- Embedding-based noun prediction
- Support for multiple semantic roles per situation

## Saved Models

Trained models are saved in the `saved_models` directory, with the best performing model saved as `best_model.pth`.

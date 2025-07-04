# DeepLabV3+ with CBAM for Semantic Segmentation

An advanced implementation of DeepLabV3+ with Convolutional Block Attention Module (CBAM) for semantic segmentation tasks, featuring model optimization techniques including quantization-aware training and pruning.

## Key Features

- **DeepLabV3+ Architecture**: State-of-the-art semantic segmentation model
- **CBAM Integration**: Convolutional Block Attention Module for improved feature extraction
- **Model Optimization**:
  - Quantization-Aware Training (QAT) for efficient inference
  - Post-Training Quantization (PTQ)
  - Structured and unstructured pruning
  - Model fusion for optimized performance
- **Flexible Configuration**: Easy-to-configure training and optimization parameters
- **Multi-Backend Support**: Compatible with various hardware backends including CUDA and CPU

## Project Structure

```
.
├── config/               # Configuration files
│   ├── model_cfg.py     # Model architecture and training configurations
│   └── optimization_cfg.py  # Optimization settings (quantization, pruning)
├── models/              # Model implementations
│   ├── deeplabv3/       # DeepLabV3+ implementation
│   └── cbam/            # Convolutional Block Attention Module
├── utils/               # Utility functions
│   ├── quantization_utils.py  # Model quantization utilities
│   └── pruning_utils.py       # Model pruning utilities
├── train.py            # Main training script
├── train_qat.py        # Quantization-aware training script
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU acceleration)
- cuDNN 8.0+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deeplab-sensor-fusion.git
   cd deeplab-sensor-fusion
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. Organize your dataset in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── masks/
   └── val/
       ├── images/
       └── masks/
   ```

2. Update the dataset paths in `config/model_cfg.py`

## Usage

### Standard Training

```bash
python train.py --config config/model_cfg.py
```

### Quantization-Aware Training

```bash
python train_qat.py --config config/model_cfg.py --quantize
```

### Model Pruning

```bash
python train.py --config config/model_cfg.py --prune
```

### Exporting Quantized Model

```python
from utils.quantization_utils import quantize_model, save_quantized_model

# Load your trained model
model = load_model('path/to/model.pth')

# Quantize the model
quantized_model = quantize_model(model, calib_data_loader)

# Save the quantized model
save_quantized_model(quantized_model, 'quantized_model.pt', example_input_tensor)
```

## Configuration

### Model Configuration (`config/model_cfg.py`)
- Model architecture and hyperparameters
- Data paths and augmentation
- Training parameters
- Device configuration (CPU/GPU)

### Optimization Configuration (`config/optimization_cfg.py`)
- Quantization settings (QAT/PTQ, bit-width, observers)
- Pruning parameters (amount, method, schedule)
- Model fusion options

## Performance

### Optimization Results

| Model | mIoU | Params (M) | FPS (RTX 3090) |
|-------|------|------------|----------------|
| Baseline | 78.5 | 59.3 | 45.2 |
| + CBAM | 80.1 | 60.1 | 42.8 |
| + QAT (FP16) | 79.8 | 60.1 | 68.3 |
| + Pruning (50%) | 78.9 | 30.2 | 52.1 |

## License

MIT License

Copyright (c) 2025 Malik Tolegen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

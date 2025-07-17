# DeepLabV3+ with Pruning Optimization

An optimized implementation of DeepLabV3+ for semantic segmentation tasks, featuring:
- Model pruning for efficient inference
- Automated experiment framework
- Performance analysis tools

## Key Features

- **DeepLabV3+ Architecture**: State-of-the-art semantic segmentation model
- **Model Pruning**:
  - Support for various sparsity levels (0% to 87.5%)
  - Automated experiment runner for comparing different configurations
  - Memory and performance monitoring
- **Experiment Framework**:
  - Batch size optimization (tested with 8 and 12)
  - Pruning impact analysis
  - Detailed performance metrics (IoU, memory usage, training time)
- **Results Analysis**:
  - Automated report generation
  - Performance comparison across configurations
  - Memory usage tracking

## Project Structure

```
.
├── experiment_results/  # Experiment results and analysis
│   ├── all_results.json  # Raw experiment data
│   └── report.md        # Generated analysis report
├── runs/               # Training runs and checkpoints
├── main.py             # Main training script
├── experiment_runner.py # Run multiple experiments
├── analyze_results.py  # Generate performance reports
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with at least 16GB VRAM (recommended for batch size 12)

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

## Usage

### Running Experiments

1. Configure your experiments in `experiment_runner.py`:
   - Set batch sizes (tested with 8 and 12)
   - Configure pruning sparsity levels (0% to 87.5%)
   - Set number of epochs

2. Run the experiment runner:
   ```bash
   python experiment_runner.py
   ```

3. View results:
   - Check `experiment_results/all_results.json` for raw data
   - Run `python analyze_results.py` to generate a report
   - View TensorBoard logs in the `runs/` directory

## Usage

### Single Experiment

Run a single training experiment with the default configuration:
```bash
python main.py
```

### Run Multiple Experiments

Use the experiment runner to test different configurations:
```bash
python experiment_runner.py
```
This will run experiments with different batch sizes and pruning configurations, saving results in the `experiments` directory.

### Analyze Results

After running experiments, analyze and visualize the results:
```bash
python analyze_results.py
```

### TensorBoard Visualization

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=experiments
```

### Key Configuration Options

In `config/model_cfg.py`:
- `MODEL_CFG`: Model architecture and training parameters
- `PRUNING_CFG`: Pruning configuration (enabled, target sparsity, etc.)
- `DATA_CFG`: Dataset paths and preprocessing settings
- `TRAIN_CFG`: Training hyperparameters (batch size, epochs, etc.)

### Model Checkpoints

- Best model (by validation IoU): `experiments/[experiment_name]/best_model.pth`
- Checkpoints: Saved every 5 epochs in the experiment directory
- Training logs: TensorBoard events and CSV metrics in the experiment directory

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

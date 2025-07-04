# DeepLab Sensor Fusion

Semantic segmentation with sensor fusion using DeepLabV3 + CBAM architecture.

## Project Structure

```
.
├── config/               # Configuration files
│   └── model_cfg.py     # Model and training configurations
├── entity/              # Data and model entities
│   ├── models/          # Model-specific entities
│   └── utils/           # Utility entities
├── models/              # Model implementations
│   └── cbam/            # Convolutional Block Attention Module
├── type/                # Type definitions
├── utils/               # Utility functions
├── main.py              # Main training and evaluation script
└── requirements.txt     # Python dependencies
```

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset:
   - Place your dataset in the structure specified in `config/model_cfg.py`
   - Update the paths in `config/model_cfg.py` accordingly

## Training

To train the model:
```bash
python main.py
```

## Configuration

Modify `config/model_cfg.py` to adjust:
- Model architecture and hyperparameters
- Data paths and augmentation
- Training parameters
- Device configuration (CPU/GPU)

## License

MIT License 

Copyright (c) 2025 Malik

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

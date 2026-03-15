# Installation Guide

## Quick Setup for Full Training Environment

To run the complete training pipeline with all features, you'll need to install the required dependencies.

### Option 1: Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate environment**:
   ```bash
   poetry shell
   ```

4. **Start training**:
   ```bash
   python train_dqn.py
   ```

### Option 2: Using pip/conda

If you prefer to use your existing Python environment:

```bash
# Core ML dependencies
pip install torch torchvision
pip install gymnasium

# Configuration and logging
pip install hydra-core omegaconf
pip install tensorboard
pip install wandb  # Optional, for advanced logging

# Visualization and interaction
pip install pygame
pip install pillow

# Development tools (optional)
pip install pytest black isort mypy
```

### Option 3: Minimal Training Setup

For a minimal setup that can run training:

```bash
pip install torch gymnasium hydra-core omegaconf
```

## Environment Compatibility

### Fixing NumPy/TensorBoard Compatibility Issues

If you encounter the `np.object` error with TensorBoard:

1. **Update packages**:
   ```bash
   pip install --upgrade numpy tensorboard
   ```

2. **Or use CPU-only PyTorch**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Or disable TensorBoard logging**:
   ```python
   # In train_dqn.py, set:
   use_tensorboard: false
   ```

### Alternative: Conda Environment

Create a clean conda environment:

```bash
conda create -n snake-rl python=3.10
conda activate snake-rl
conda install pytorch torchvision torchaudio -c pytorch
pip install gymnasium hydra-core omegaconf tensorboard
```

## Verification

Test your installation:

```bash
# Test core engine (no dependencies required)
python demo_engine.py

# Test training components (requires PyTorch)
python train_simple.py

# Full training (requires all dependencies)
python train_dqn.py training.total_steps=1000
```

## What Works Without Dependencies

Even without installing additional packages, you can:

- ✅ Run the high-performance Snake engine (`demo_engine.py`)
- ✅ Test all core components (`test_complete_pipeline.py`)
- ✅ Explore the codebase and architecture
- ✅ Understand the training pipeline structure

## Troubleshooting

### Common Issues

1. **PyTorch CUDA issues**: Use CPU version if you don't have CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Gymnasium installation issues**:
   ```bash
   pip install gymnasium[classic_control]
   ```

3. **Hydra configuration errors**: Make sure you're in the project root directory when running training scripts.

4. **Import errors**: Ensure the project root is in your Python path or install in development mode:
   ```bash
   pip install -e .
   ```

### Performance Notes

- **Training speed**: GPU recommended for faster training, but CPU works fine for experimentation
- **Memory usage**: Reduce `buffer_size` if you encounter memory issues
- **FPS**: The engine achieves >450K FPS as demonstrated, well above the 5K target

## Next Steps

Once installed, try:

1. **Quick training run**:
   ```bash
   python train_dqn.py env=small training.total_steps=10000
   ```

2. **Evaluate a model**:
   ```bash
   python evaluate.py checkpoints/best_model.pt --episodes 10
   ```

3. **Interactive demo**:
   ```bash
   python demo.py --model checkpoints/best_model.pt
   ```

## Development Setup

For development work:

```bash
# Install with development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black snake_rl/
isort snake_rl/
```

The project is designed to work with or without external dependencies, so you can explore the codebase and understand the architecture even before installing the full ML stack.
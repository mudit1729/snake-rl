#!/usr/bin/env python3
"""
Snake-from-Pixels RL Project Environment Setup
Generates pyproject.toml with Poetry configuration and all dependencies.
"""

import subprocess
import sys
from pathlib import Path

def create_pyproject_toml():
    """Generate pyproject.toml with all required dependencies."""
    
    pyproject_content = '''[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "snake-rl"
version = "0.1.0"
description = "Snake-from-Pixels Reinforcement Learning Project"
authors = ["Snake RL Team"]
license = "MIT"
readme = "README.md"
packages = [{include = "snake_rl"}]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
numpy = "^1.24.0"
torch = ">=2.1.0"
torchvision = ">=0.16.0"
gymnasium = "^0.29.0"
pygame = "^2.5.0"
tensorboard = "^2.15.0"
wandb = "^0.16.0"
hydra-core = "^1.3.0"
opencv-python = "^4.8.0"
pillow = "^10.0.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
tqdm = "^4.66.0"
omegaconf = "^2.3.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.7.0"
isort = "^5.12.0"
mypy = "^1.5.0"
pre-commit = "^3.3.0"
flake8 = "^6.0.0"
jupyter = "^1.0.0"
ipykernel = "^6.25.0"

[tool.black]
line-length = 88
target-version = ['py310']
include = '\\.pyi?$'
extend-exclude = """
/(
  # directories
  \\.eggs
  | \\.git
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
"""

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["snake_rl"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "pygame.*",
    "wandb.*",
    "tensorboard.*",
    "cv2.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["snake_rl/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=snake_rl",
    "--cov-branch",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
]

[tool.coverage.run]
source = ["snake_rl"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
'''

    # Write pyproject.toml
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    print("✅ Created pyproject.toml with Poetry configuration")
    
    # Create pre-commit configuration
    precommit_content = '''repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-case-conflict
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]
        args: ['--max-line-length=88', '--extend-ignore=E203,W503,D100,D101,D102,D103,D104,D105,D106,D107']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]
'''
    
    with open(".pre-commit-config.yaml", "w") as f:
        f.write(precommit_content)
    
    print("✅ Created .pre-commit-config.yaml")

def check_poetry_installation():
    """Check if Poetry is installed and install if necessary."""
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poetry is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Poetry not found. Please install Poetry first:")
    print("   curl -sSL https://install.python-poetry.org | python3 -")
    print("   Or visit: https://python-poetry.org/docs/#installation")
    return False

def setup_environment():
    """Set up the complete development environment."""
    print("🐍 Setting up Snake-from-Pixels RL development environment...")
    print()
    
    # Create pyproject.toml and configs
    create_pyproject_toml()
    
    # Check Poetry installation
    if not check_poetry_installation():
        print("\n⚠️  Poetry installation required to continue.")
        print("After installing Poetry, run:")
        print("  poetry install")
        print("  poetry run pre-commit install")
        return False
    
    # Install dependencies
    print("\n📦 Installing dependencies with Poetry...")
    try:
        subprocess.run(["poetry", "install"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Set up pre-commit hooks
    print("\n🔧 Setting up pre-commit hooks...")
    try:
        subprocess.run(["poetry", "run", "pre-commit", "install"], check=True)
        print("✅ Pre-commit hooks installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to install pre-commit hooks: {e}")
        print("You can install them later with: poetry run pre-commit install")
    
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("  1. Activate the environment: poetry shell")
    print("  2. Run tests: poetry run pytest")
    print("  3. Check linting: poetry run pre-commit run --all-files")
    print("  4. Start implementing the snake simulator")
    
    return True

if __name__ == "__main__":
    setup_environment()
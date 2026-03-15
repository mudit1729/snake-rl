#!/bin/bash

# Snake-from-Pixels RL Project Initialization Script
set -e

echo "🐍 Initializing Snake-from-Pixels RL Project..."

# Create main project directories
mkdir -p snake_rl/{sim,env,algos/{models},experiments,tests,docs}
mkdir -p snake_rl/conf

echo "📁 Created directory structure:"
echo "  snake_rl/"
echo "    ├── sim/          # Core snake simulator"
echo "    ├── env/          # Gymnasium environment wrappers"  
echo "    ├── algos/        # RL algorithms"
echo "    │   └── models/   # Neural network models"
echo "    ├── experiments/  # Training scripts and sweeps"
echo "    ├── tests/        # Unit tests"
echo "    ├── docs/         # Documentation"
echo "    └── conf/         # Hydra configuration files"

# Initialize Git repository
if [ ! -d ".git" ]; then
    git init
    echo "📦 Initialized Git repository"
fi

# Create MIT License
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Snake-from-Pixels RL Project

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
EOF

# Create .gitignore for Python/PyTorch projects
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# PyTorch specific
*.pth
*.pt
checkpoints/
runs/
logs/

# Weights & Biases
wandb/

# Hydra outputs
outputs/
multirun/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE specific
.vscode/
.idea/
*.swp
*.swo
*~

# Temporary files
*.tmp
*.temp
EOF

# Create initial __init__.py files
touch snake_rl/__init__.py
touch snake_rl/sim/__init__.py
touch snake_rl/env/__init__.py
touch snake_rl/algos/__init__.py
touch snake_rl/algos/models/__init__.py
touch snake_rl/experiments/__init__.py
touch snake_rl/tests/__init__.py

echo "📄 Created LICENSE and .gitignore"
echo "🔧 Created __init__.py files for Python modules"
echo "✅ Repository initialization complete!"
echo ""
echo "Next steps:"
echo "  1. Run setup_env.py to create pyproject.toml"
echo "  2. Set up CI/CD with GitHub Actions"
echo "  3. Start implementing the core snake simulator"
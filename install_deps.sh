#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Install dependencies
pip install torch torchvision "joblib== 1.2.0" ipython matplotlib seaborn pandas scikit-learn scipy tqdm
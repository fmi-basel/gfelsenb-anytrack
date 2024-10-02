#!/bin/zsh
### Aug 8, 2024
# Test run
#
eval "$(conda shell.bash hook)"
conda activate tracking
poetry run python setup.py install --user
poetry run python run.py

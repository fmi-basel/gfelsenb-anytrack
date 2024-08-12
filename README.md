# anytrack
Pipeline to track individual objects in videos with static backgrounds.

## Getting started

### Installation

Install anytrack using the repository setup file. First, clone the repository
```bash
git clone https://github.com/Felsenberg-lab/anytrack.git && cd anytrack
```
Install Python 3.12 using Anaconda
```bash
conda env create -n anytrack -f tracking_environment.yml
```
Now, install pipx as described on the [website](https://pipx.pypa.io/stable/installation/). Next, install poetry 
```bash
pipx install poetry
```
Finally, install required pip packages via Poetry and install this package to your system
```bash
poetry install && poetry run python setup.py install --user
```
This finishes the installation process for this package.

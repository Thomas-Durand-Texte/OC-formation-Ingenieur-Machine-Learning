#!/bin/usr/bash

sudo dnf install python3.8

if [ ! -e ".venv" ]
then
echo "Creation of .venv"
python3.8 -m venv .venv
fi

source .venv/bin/activate
if [[ "$VIRTUAL_ENV" == "" ]]
then
  echo ""
  echo "!!! Problem with virtual environment !!!"
  echo ""
  exit
fi

# pip uninstall numpy ## only if numpy is already installed
# pip uninstall scipy ## only if scipy is already installed

# export LAPACK=/usr/lib64/liblapack.so
# export ATLAS=/usr/lib64/atlas/libtatlas.so
# export BLAS=/usr/lib64/libblas.so



pip install --upgrade pip
pip install ipykernel -U --force-reinstall
python3 -m install jupyer
pip install numpy scipy matplotlib pandas  # dask

pip install -U numpy scipy matplotlib scikit-learn

pip install seaborn missingno statsmodels plotly  # ipywidgets


# pytorch
pip3 install torch torchvision torchaudio torchview torchviz  # torchviz hiddenlayer # --index-url https://download.pytorch.org/whl/cu118

# TENSORFLOW
# Current stable release for CPU and GPU
# pip install tensorrt
# pip install tensorflow
# pip install tensorflow_hub

# Or try the preview build (unstable)
# pip install tf-nightly

deactivate

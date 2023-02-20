#!/bin/bash

# FAIRE UN TEST SI .VENV EXISTS

if [ ! -e ".venv" ]
then
echo "Creation of .venv"
python3 -m venv .venv
fi

source .venv/bin/activate

# pip uninstall numpy ## only if numpy is already installed
# pip uninstall scipy ## only if scipy is already installed

# export LAPACK=/usr/lib64/liblapack.so
# export ATLAS=/usr/lib64/atlas/libtatlas.so
# export BLAS=/usr/lib64/libblas.so


pip3 install --upgrade pip
pip install ipykernel -U --force-reinstall
pip3 install numpy scipy matplotlib pandas seaborn missingno statsmodels plotly pingouin lightgbm # dask

pip3 install -U numpy scipy matplotlib scikit-learn

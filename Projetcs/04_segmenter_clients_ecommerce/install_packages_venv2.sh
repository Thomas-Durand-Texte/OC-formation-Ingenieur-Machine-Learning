#!/bin/usr/bash

# FAIRE UN TEST SI .VENV EXISTS

if [ ! -e ".venv2" ]
then
echo "Creation of .venv2"
python3.10 -m venv .venv2
fi

source .venv2/bin/activate

# pip uninstall numpy ## only if numpy is already installed
# pip uninstall scipy ## only if scipy is already installed

# export LAPACK=/usr/lib64/liblapack.so
# export ATLAS=/usr/lib64/atlas/libtatlas.so
# export BLAS=/usr/lib64/libblas.so


pip install --upgrade pip
pip install ipykernel -U --force-reinstall
pip install --upgrade nbformat

pip install numpy scipy matplotlib pandas # dask

pip install -U numpy scipy matplotlib scikit-learn

pip install seaborn missingno statsmodels plotly pingouin lightgbm shap folium skops yellowbrick # dask

pip install k-means-constrained

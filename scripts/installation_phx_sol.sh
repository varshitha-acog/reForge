module load mamba
mamba env create -n reforge --file environment.yml
source activate reforge
pip install -e .
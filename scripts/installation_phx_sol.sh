module load mamba/latest  
mamba env create -n reforge --file environment.yml
source activate reforge
pip install -e .
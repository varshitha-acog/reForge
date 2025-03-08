#!/bin/bash                     
# IMPORTANT! FOR PHX CLUSTER USERS! 
# RUN THE SCRIPT as ". phx_md_load.sh"
# with a dot (.) commands run in the current shell environment.
# NOT "bash phx_md_load.sh"
# "bash" starts a new shell process to execute the script.

module purge
module load mamba/latest
module load gromacs-2023.3-openmpi-cuda-qx
source activate reforge 


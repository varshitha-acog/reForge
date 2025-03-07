#!/usr/bin/env python
"""
Hello World!
=================

Interface Tutorial

Requirements:
    - HPC cluster
    - Python 3.x

Author: DY
"""

from pathlib import Path
import os
import time

#%%
# One of the motivations behind reForge was to provide a user- and beginner-friendly interface
# for managing potentially hundreds or thousands of MD simulations without having to switch 
# between multiple scripts and constantly rewriting them—all while staying within comfort of Python.
# This is what the 'cli' module is for:
from reforge.cli import run, sbatch, dojob

#%%
# The idea is very simple: suppose you have "n" mutants and you need to perform "n" independent runs 
# for each mutant to achieve sufficient sampling.
systems = [f'mutant_{i}' for i in range(4)]
mdruns = [f'mdrun_{i}' for i in range(4)]

#%%
# The functions run, sbatch, and dojob allow you to execute the same script for all cases:
for system in systems:
    for mdrun in mdruns:
        run(f'echo "Hello, I\'m run {mdrun} of {system}"')

#%%
# Alternatively, you can call run like this:
for system in systems:
    for mdrun in mdruns:
        run('echo', 'Hello, I\'m run', f'{mdrun} of {system}')     

#%%
# Alternatively, you can sbatch a script. This command passes all positional arguments to the script 
# and keyword arguments to the SLURM scheduler.
if 0:  # Set to 1 to actually submit jobs.
    for system in systems:
        for mdrun in mdruns:
            sbatch('hello.sh', mdrun, system, mem='50M', t='00:00:01', partition='htc', qos='public')   

#%%         
# By default, output is written to "slurm_jobs/output_file" and "slurm_jobs/error_file".
outfiles = [f for f in Path('slurm_jobs').glob('*output*')]
for fpath in outfiles:
    with open(fpath, 'r') as file:
        # Print only the first line of each output file.
        print(file.readline(), end='')

#%%
# For convenience, dojob will either submit or run the script based on its first argument.
# In this case, we do not want to sbatch the script.
submit = False
for system in systems:
    for mdrun in mdruns:
        dojob(submit, 'hello.sh', mdrun, system, mem='50M', t='00:00:01', partition='htc', qos='public')

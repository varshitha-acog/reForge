#!/bin/bash
#SBATCH --time=0-01:00:00                                                       
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH -o tests/sl_output.out
#SBATCH -e tests/sl_error.err

echo "Hello I am run $1 of system $2"
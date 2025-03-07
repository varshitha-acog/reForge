import os
import numpy as np
import pandas as pd
import sys
from reforge import io
from reforge.mdsystem import gmxmd
from reforge.utils import logger
from reforge.mdm import percentile


def set_bfactors_by_residue(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    residues = atoms.residues
    for idx, residue in enumerate(residues):
        for atom in residue:
            atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def make_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pdbdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pdbdir, f'{label}_err.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)
    set_bfactors_by_residue(system.inpdb, err, err_pdb)


def make_delta_pdb(system_1, system_2, label, out_name, multiply_by_len=False):
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if multiply_by_len:
        data_1 *= len(data_1)
        err_1 *= len(err_1)
        data_2 *= len(data_2)
        err_2 *= len(err_2)  
    data = data_1 - data_2
    err = 0.5 * np.sqrt(err_1**2 + err_2**2)
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)
    set_bfactors_by_residue(system_1.inpdb, err, err_pdb) 


def dfi_pdb(system):
    logger.info(f'Making DFI PDB')
    make_pdb(system, 'dfi', multiply_by_len=True)


def dci_pdbs(system):
    for chain in system.segments:
        logger.info(f'Making DCI {chain} PDB')
        make_pdb(system, label=f'gdci_{chain}')
        make_pdb(system, label=f'gtdci_{chain}')
        make_pdb(system, label=f'gasym_{chain}')

 
if __name__ == "__main__":
    sysdir = 'systems' 
    sysname = 'egfr_go'
    system = gmxmd.GmxSystem(sysdir, sysname)
    make_pdb(system, 'dfi')
    make_pdb(system, 'rmsf')
    dci_pdbs(system)


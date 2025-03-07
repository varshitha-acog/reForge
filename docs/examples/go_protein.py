#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup

Requirements:
    - GROMACS
    - Python 3.x

Author: DY
"""

import os
from reforge import cli
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

#%%
# First, we need to initialize an instance of *GmxSystem*, which will handle path management
# and the necessary files. This instance uses the parent directory *'systems'* (relative to the current
# directory) and the root directory *'test'* within *'systems'* for the system. 
mdsys = GmxSystem(sysdir='systems', sysname='test')

#%%
# Next, prepare the necessary file and directories by calling *prepare_files()*.
mdsys.prepare_files()

# List the files in the system's root directory:
for f in mdsys.root.iterdir():
    print(f)

#%%
# Sort chains and atoms in the input PDB file to avoid future conflicts.
# This creates a file (named *inpdb.pdb*) that can later be accessed as *mdsys.inpdb*.
in_pdb = mdsys.root / "1btl.pdb"  # can be relative to *mdsys.root* or an absolute path
mdsys.sort_input_pdb(in_pdb)
print(mdsys.inpdb)

#%%
# Although there are multiple chains in this case, the *split_chains()* method
# splits *mdsys.inpdb* into separate chains and moves protein and RNA/DNA files to their respective directories.
mdsys.split_chains()

#%%
# Coarse-grain the proteins using *martinize2* (by Martini):
mdsys.martinize_proteins_go(go_eps=10.0, go_low=0.3, go_up=1.0, p="backbone", pf=500, append=False)

#%%
# Inspect the generated topology files. The topology should include the main protein topology file
# *chain_A.itp* as well as the virtual sites parameters for the GO model:
# *go_atomtypes.itp* and *go_nbparams.itp*.
for f in mdsys.topdir.iterdir():
    print(f)

#%% 
# Check the coarse-grained structure in the *cgdir* directory.
for f in mdsys.cgdir.iterdir():
    print(f)

#%% 
# Combine all topology and structure files.
# The method *make_cg_topology()* uses GROMACS's *gmx pdb2gmx* module to create the simulation box.
# (See online documentation for details.)
mdsys.make_cg_topology() # It returns the CG topology as *mdsys.systop* (i.e. "system.top").
mdsys.make_cg_structure(bt='dodecahedron', d='1.2') # Returns *mdsys.solupdb* (i.e. "solute.pdb").

#%% 
# Add solvent and neutralize the system's charge.
solvent = mdsys.root / "water.gro"
mdsys.solvate(cp=mdsys.solupdb, cs=solvent)
mdsys.add_bulk_ions(conc=0.15, pname="NA", nname="CL")

#%% 
# To work with GROMACS selections, generate an index file (*mdsys.sysndx*).
# The default group order is: 1. *System*, 2. *Solute*, 3. *Backbone*, 4. *Solvent*, 5. *Not Water*, and then 6+ individual chains.
# Custom groups can be added using the method *AtomList.write_to_ndx()*.
mdsys.make_system_ndx(backbone_atoms=["BB"])

#%%
# One of the convenient features of *GmxSystem* is the ability to execute GROMACS commands
# directly from your Python script using *GmxSystem.gmx*. This runs the command in the system's working directory.
# For example, to view the system (using VMD or PyMOL), you must first correct the generated box for
# periodic boundary conditions. This is done with GROMACS's *trjconv* module:
mdsys.gmx("trjconv", clinput='0\n', s=mdsys.syspdb, f=mdsys.syspdb, pbc='atom', ur='compact', o="viz.pdb")

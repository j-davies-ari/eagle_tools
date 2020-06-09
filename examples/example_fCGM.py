# -*- coding: utf-8 -*-

import numpy as np
import eagle_tools

# Your group numbers here - this example will just compute 10 test haloes
groupnumbers = np.arange(1000,1009,dtype=np.int64)

# Initialise snapshot object
Snapshot = eagle_tools.read.snapshot(sim='L0100N1504',run='REFERENCE',tag='028_z000p000')

# Initialise output array
f_CGM = np.empty(len(groupnumbers))

# Loop over group numbers
for g, gn in enumerate(groupnumbers):

    # Select all particles within r200
    Snapshot.select(gn,parttype=0,region_size='r200')

    # Load in particle quantities
    mass = Snapshot.load('Mass')
    sfr = Snapshot.load('StarFormationRate') # My definition excludes SFing gas, so I need this

    # Compute f_CGM normalised by the cosmic baryon fraction -----> Mgas/M200 / f_b_cosmic
    f_CGM[g] = (np.sum(mass[sfr==0.])/Snapshot.this_M200) / (Snapshot.header['OmegaBaryon']/Snapshot.header['Omega0'])

print f_CGM
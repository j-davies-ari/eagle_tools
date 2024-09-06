import numpy as np
import h5py as h5

import eagle_tools

# Output file
f = h5.File("f_CGM.hdf5",'w')

# Initialise the snapshot
snap = eagle_tools.read.Snapshot(
        sim = "L0100N1504",
        model = "REFERENCE",
        tag = '028_z000p000',
        data_location = "/mnt/aridata1/projects/EAGLE"
    )

# Load the halo mass catalogue
snap.load_subfind('FOF/Group_M_Crit200')
m200_catalogue = snap.subfind['FOF/Group_M_Crit200']

# Just do haloes with M200>10^10 for now
groupnumbers_to_do = np.where(m200_catalogue*1e10 > np.power(10.,11.5))[0] + 1 

# Make arrays to store the results in
n_gns = len(groupnumbers_to_do)
f_CGM_all = np.empty(n_gns, dtype=np.float64) 
f_CGM_no_sf = np.empty(n_gns, dtype=np.float64) 
f_b = np.empty(n_gns, dtype=np.float64) 

# Get the cosmic baryon fraction for normalisation later
f_b_cosmic = snap.header['OmegaBaryon']/snap.header['Omega0']

for g, gn in enumerate(groupnumbers_to_do):

    print(f"Doing halo {g} of {n_gns}",flush=True)

    # Get the halo mass
    m200 = m200_catalogue[gn-1]

    # All gas within r200
    halo_gas = snap.select_halo(gn,0,parttype=0,radius='FOF/Group_R_Crit200')

    # All stars within r200
    halo_stars = snap.select_halo(gn,0,parttype=4,radius='FOF/Group_R_Crit200')

    mass_gas = snap.load(halo_gas,'Mass')
    oes = snap.load(halo_gas,'OnEquationOfState') # so we can exclude the ISM
    mass_stars = snap.load(halo_stars,'Mass')

    f_CGM_all[g] = (np.sum(mass_gas)/m200)/f_b_cosmic

    f_CGM_no_sf[g] = (np.sum(mass_gas[oes<=0.])/m200)/f_b_cosmic # Gas that is not currently on the equation of state (i.e. in the ISM)

    f_b[g] = ((np.sum(mass_gas)+np.sum(mass_stars))/m200)/f_b_cosmic

print('Saving...')

f.create_dataset('GroupNumber',data=groupnumbers_to_do)
f.create_dataset('f_CGM_all',data=f_CGM_all)
f.create_dataset('f_CGM_no_sf',data=f_CGM_no_sf)
f.create_dataset('f_b',data=f_b)

f.close()
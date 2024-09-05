"""
Does the same job as f_CGM.py, but splits the haloes across MPI ranks.
This is an example of how you can parallelise halo-by-halo tasks easily with the select_halo workflow.
"""

import numpy as np
import h5py as h5
from mpi4py import MPI

import eagle_tools

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

# Set up the MPI communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initialise the snapshot on all ranks
snap = eagle_tools.read.Snapshot(
        sim = "L0100N1504",
        model = "REFERENCE",
        tag = '028_z000p000',
        data_location = "/mnt/aridata1/projects/EAGLE"
    )

# Load the halo mass catalogue on all ranks
snap.load_subfind('FOF/Group_M_Crit200')
m200_catalogue = snap.subfind['FOF/Group_M_Crit200']

# Dummy variable to hold the groupnumbers after they are scattered between ranks
split_gns = None

if rank == 0:

    f = h5.File("f_CGM.hdf5",'w')

    # Split up the group numbers. Don't want to retain the ordering as we want to distribute the most massive objects
    # between cores, as they take the most time
    # Just do haloes with M200>10^10 for now
    
    groupnumbers_to_do = np.where(m200_catalogue*1e10 > np.power(10.,11.5))[0] + 1

    split_gns = split(groupnumbers_to_do,size)


# Scatter jobs across cores.
split_gns = comm.scatter(split_gns, root=0)

print('Rank: ',rank, ', recvbuf received: ',split_gns)

# Make arrays to store the results in
numDataPerRank = len(split_gns)
f_CGM_all = np.empty(numDataPerRank, dtype=np.float64) 
f_CGM_no_sf = np.empty(numDataPerRank, dtype=np.float64) 
f_b = np.empty(numDataPerRank, dtype=np.float64) 

# Get the cosmic baryon fraction for normalisation later
f_b_cosmic = snap.header['OmegaBaryon']/snap.header['Omega0']

for g, gn in enumerate(split_gns):

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

# Gather up the results onto rank 0
out_gns = comm.gather(split_gns, root=0)
out_f_CGM_all = comm.gather(f_CGM_all, root=0)
out_f_CGM_no_sf = comm.gather(f_CGM_no_sf, root=0)
out_f_b = comm.gather(f_b, root=0)

if rank == 0:

    # Concatenate all the split arrays received from the MPI ranks and sort by group number
    out_gns = np.concatenate(out_gns,axis=0)
    gnsort = np.argsort(out_gns)
    out_gns = out_gns[gnsort]

    out_f_CGM_all = np.concatenate(out_f_CGM_all,axis=0)[gnsort]
    out_f_CGM_no_sf = np.concatenate(out_f_CGM_no_sf,axis=0)[gnsort]
    out_f_b = np.concatenate(out_f_b,axis=0)[gnsort]

    print('Saving...')

    f.create_dataset('GroupNumber',data=out_gns)
    f.create_dataset('f_CGM_all',data=out_f_CGM_all)
    f.create_dataset('f_CGM_no_sf',data=out_f_CGM_no_sf)
    f.create_dataset('f_b',data=out_f_b)

    f.close()
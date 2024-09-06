import numpy as np


def soft_comov(z,sim): # Returns the comoving softening length for any redshift in the EAGLE boxes
    a = 1./(1.+z)
    if sim=='L0025N0376' or sim=='L0050N0752' or sim=='L0100N1504':
        e_com = 2.66
        e_max_phys = 0.7
    elif sim == 'L0025N0752':
        e_com = 1.33
        e_max_phys = 0.35
    else:
        raise NameError('Softening for this simulation not known yet!')
    if e_com < e_max_phys/a:
        e = e_com
    else:
        e = e_max_phys/a
    return e

def soft_phys(z,sim):
    ecom = soft_comov(z,sim)
    return 1./(1.+z) * ecom

def get_binedges(z_store): # Creates an array of bin edges for variable-width histogram plotting, given the bin centres
    bins = []
    bins.append(z_store[0]-(z_store[1]-z_store[0])/2)
    for j in range(len(z_store)):
        if j+1 == len(z_store):
            break
        else:
            bins.append(z_store[j]+(z_store[j+1]-z_store[j])/2)
    bins.append(z_store[-1]+(z_store[-1]-z_store[-2])/2)
    
    return bins
    
def get_bincentres(binedges): # Finds the centre points of a set of bin edges
    bincentres = []
    for i in range(len(binedges)):
        if i+1 == len(binedges):
            break
        else:
            bincentres.append((binedges[i+1]+binedges[i])/2.)
    return np.array(bincentres)
    
def get_binsizes(binedges):
    binsizes = []
    for i in range(len(binedges)):
        if i+1 == len(binedges):
            break
        else:
            binsizes.append(binedges[i+1]-binedges[i])
    return binsizes

class Constants(object):
    '''
    Useful constants for working with EAGLE data
    '''

    unit_mass_cgs = np.float64(1.989e43)
    unit_time_cgs = 3.085678E19
    unit_density_cgs = 6.769911178294543e-31
    unit_velocity_cgs = 100000.
    unit_energy_cgs = 1.989E53
    unit_length_cgs = 3.085678E24
    unit_pressure_cgs = 6.769911178294542E-21

    m_sol_SI = 1.989e30
    m_sol_cgs = 1.989e33

    Mpc_SI = 3.0856e16 * 1e6
    Mpc_cgs = Mpc_SI * 100.
    G_SI = 6.6726e-11
    G_cgs = 6.6726e-8

    Gyr_s = 3.1536e16
    year_s = Gyr_s/1e9

    c_SI = 299792458.
    c_CGS = c_SI*100.

    BHAR_cgs = 6.445909132449984e23
    BH_erg_per_g = (0.1 * 0.15 * c_CGS**2) # eps_r eps_f c**2
    SN_erg_per_g = 8.73e15

    m_H_cgs = 1.6737e-24
    m_p_cgs = 1.6726219e-24

    Z_sol = 0.0127

    boltzmann_cgs = np.float64(1.38064852e-16)
    boltzmann_eV_per_K = np.float64(8.61733035e-5)

    thompson_cgs = 6.65245e-25
    ergs_per_keV = 1.6021773e-9

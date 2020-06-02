import numpy as np

class constants(object):
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

    # Don't use hard-wired cosmological parameters
    # h = 0.6777
    # omega_m_planck = 0.307
    # omega_vac_planck = 0.693
    # omega_b_planck = 0.04825
    # f_b_universal_planck = omega_b_planck/omega_m_planck

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

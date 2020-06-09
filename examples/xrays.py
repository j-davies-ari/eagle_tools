# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import eagle_tools
from tqdm import tqdm
import h5py as h5


# Initialise the emission module with the file path and our energy band, in this case 0.5-2.0 keV.

apec = eagle_tools.emission.apec('/hpcdata0/arijdav1/APEC_cooling_tables/APEC_spectra_0.02_80.0keV_res_10eV_interp.hdf5',energy_band=[0.5,2.])


############################################################################################################################################
# Extract spectra for H, Ne and Fe for a narrow range of temperatures, to demonstrate the interpolation

# Particle temperatures between 10^6.9 and 10^7.1 K
temperatures = np.logspace(6.9,7.1,10)
alpha = np.linspace(1.,0.,10)

# Get the spectra for those temperatures. Each row of the output array will contain a spectrum.
H_spectra = apec.get_spectra('HYDROGEN',temperatures)
Ne_spectra = apec.get_spectra('NEON',temperatures)
Fe_spectra = apec.get_spectra('IRON',temperatures)

plt.figure()

for t, T in enumerate(temperatures):

    if t == 0:
        plt.plot(apec.energy_bins,np.log10(H_spectra[t,:]) - np.log10(apec.Ebinwidth),c='navy',alpha=alpha[t],label='H') 
        plt.plot(apec.energy_bins,np.log10(Ne_spectra[t,:]) - np.log10(apec.Ebinwidth),c='green',alpha=alpha[t],label='Ne') 
        plt.plot(apec.energy_bins,np.log10(Fe_spectra[t,:]) - np.log10(apec.Ebinwidth),c='coral',alpha=alpha[t],label='Fe')
    else:
        plt.plot(apec.energy_bins,np.log10(H_spectra[t,:]) - np.log10(apec.Ebinwidth),c='navy',alpha=alpha[t]) 
        plt.plot(apec.energy_bins,np.log10(Ne_spectra[t,:]) - np.log10(apec.Ebinwidth),c='green',alpha=alpha[t]) 
        plt.plot(apec.energy_bins,np.log10(Fe_spectra[t,:]) - np.log10(apec.Ebinwidth),c='coral',alpha=alpha[t])

plt.annotate(r'$T=10^{6.9-7.1}\,\mathrm{K}$',xy=(2.,-22.5),fontsize=12)

plt.xlim(0.,3.)
plt.ylim(-27.,-21.)
plt.ylabel(r'$\log\left(d\lambda/dE_{\gamma}\right)\,[\mathrm{erg}\,\mathrm{cm}^{3}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}]$',fontsize=16)
plt.xlabel(r'$E_{\gamma}$ $[\mathrm{keV}]$',fontsize=16)
plt.legend(loc='upper right')
plt.show()






############################################################################################################################################
# Compute the X-ray luminosities of all haloes with M200>10^11.5 Msun

# Output file
f = h5.File('/hpcdata0/arijdav1/xray_test.hdf5','w')


# Initialise snapshot object
Snapshot = eagle_tools.read.snapshot(sim='L0100N1504',run='REFERENCE',tag='028_z000p000')

# Make mass selection
sample = np.where(Snapshot.M200*1e10>np.power(10.,11.5))[0]

# Get group info
M200 = Snapshot.M200[sample]
groupnumbers = Snapshot.groupnumbers[sample]

# Initialise output array
Lx_halo = np.empty(len(groupnumbers))

# Loop over group numbers
for g in tqdm(range(len(groupnumbers))):

    # Select all particles within r200
    Snapshot.select(groupnumbers[g],parttype=0,region_size='r200')

    # Load in particle quantities
    mass = Snapshot.load('Mass',cgs_units=True)
    density = Snapshot.load('Density',cgs_units=True)
    temperature = Snapshot.load('Temperature')
    abundances, num_ratios, Xe, Xi, mu = Snapshot.load_abundances()

    # My definition excludes SFing gas, so I need this
    sfr = Snapshot.load('StarFormationRate') 
    nosf = np.where(sfr==0.)[0]

    # Compute X-ray luminosities using emission module
    Lx_particles = apec.xray_luminosity(temperature[nosf],density[nosf],mass[nosf],abundances[nosf])

    Lx_halo[g] = np.sum(Lx_particles)


f.create_dataset('M200',data=M200)

f.create_dataset('Lx',data=Lx_halo)

f.close()


fig, ax = plt.subplots(1)

ax.scatter(np.log10(M200*1e10),np.log10(Lx_halo),edgecolors='b',s=5,facecolors='none',rasterized=True)

plt.show()
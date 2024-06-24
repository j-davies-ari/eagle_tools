# -*- coding: utf-8 -*-

import numpy as np

import h5py as h5
from sys import exit
from importlib import import_module
from astropy.cosmology import FlatLambdaCDM
from copy import deepcopy

class snapshot(object):

    def __init__(self,sim = 'L0100N1504',
                        run = 'REFERENCE',
                        tag='028_z000p000',
                        pdata_type = 'SNAPSHOT',
                        data_location = '/hpcdata0/simulations/EAGLE/'):

        self.read = import_module('pyread_eagle')
        # self.read = import_module('read_eagle')

        self.sim = sim
        self.run = run
        self.tag = tag
        self.pdata_type = pdata_type
        self.sim_path = data_location + sim + '/' + run + '/data/'

        # Create strings to point the read module to the first snapshot/particle/subfind files
        if pdata_type == 'SNAPSHOT':
            self.snapfile = self.sim_path + 'snapshot_'+tag+'/snap_'+tag+'.0.hdf5'
        elif pdata_type == 'PARTDATA':
            self.snapfile = self.sim_path + 'particledata_'+tag+'/eagle_subfind_particles_'+tag+'.0.hdf5'
        else:
            raise TypeError('Please pick a valid pdata_type (SNAPSHOT or PARTDATA)')
        self.subfindfile = self.sim_path + 'groups_'+tag+'/eagle_subfind_tab_' + tag + '.'

        # Get volume information. Note that header is taken from one file only, so quantities such as 'NumPart_ThisFile' are not useful.
        self.header = self.attrs('Header')

        # Assign some of the most useful quantities to the snapshot object
        self.NumPart = self.header['NumPart_Total']
        self.h = self.header['HubbleParam']
        self.aexp = self.header['ExpansionFactor']
        self.z = self.header['Redshift']
        self.masstable = self.header['MassTable']/self.h
        boxsize = self.header['BoxSize']
        self.physical_boxsize = boxsize * self.aexp/self.h
        self.f_b_cosmic = self.header['OmegaBaryon']/self.header['Omega0']

        # Create an astropy.cosmology object for doing cosmological calculations.

        self.cosmology = FlatLambdaCDM(100.*self.header['HubbleParam'],Om0=self.header['Omega0'],Ob0=self.header['OmegaBaryon'])

        # Load in basic catalogues for halo/subhalo identification

        # first_subhalo = np.array(self.E.readArray("SUBFIND_GROUP", self.sim_path, self.tag, 'FOF/FirstSubhaloID'))
        self.first_subhalo = self.fof('FirstSubhaloID')

        # If the final FOF group is empty, it can be assigned a FirstSubhaloID which is out of range
        self.subhalo_COP = self.subfind('CentreOfPotential')
        max_subhalo = len(self.subhalo_COP[:,0])
        self.first_subhalo[self.first_subhalo==max_subhalo] -= 1 # this line fixes the issue

        # Group number and subgroup number for all subhaloes
        # self.subhalo_gn = self.subfind('GroupNumber')
        # self.subhalo_sgn = self.subfind('SubGroupNumber')
        # self.subhalo_COP = self.subfind('CentreOfPotential')
        # self.subhalo_bulk_velocity = self.subfind('Velocity')

        # Useful quantities for centrals only
        # self.central_Mstar_30kpc = self.subhalo_Mstar_30kpc[self.first_subhalo]
        self.central_COP = self.subhalo_COP[self.first_subhalo,:]
        # self.central_bulk_velocity = self.subhalo_bulk_velocity[self.first_subhalo,:]
        self.r200 = self.fof('Group_R_Crit200')
        self.M200 = self.fof('Group_M_Crit200')
        # self.nsub = self.fof('NumOfSubhalos')
        self.groupnumbers = np.arange(len(self.M200)) + 1

        self.have_run_select = False # This is set to True when a region has been selected - prevents crashes later on.
        

    def catalogue_read(self,quantity,table,phys_units=True,cgs_units=False,verbose=False):
        '''
        Read in FOF or SUBFIND catalogues.
        The user may prefer to use the wrapper functions 'fof' and 'subfind' rather than specifying 'table' here.
        '''

        assert table in ['FOF','Subhalo'],'table must be either FOF or Subhalo'

        file_ind = 0
        while True:

            try:
                with h5.File(self.subfindfile+str(file_ind)+'.hdf5', 'r') as f:
                    
                    if file_ind == 0:
                        
                        data = f['/'+table+'/%s'%(quantity)]

                        # Grab conversion factors from first file
                        h_conversion_factor = data.attrs['h-scale-exponent']
                        aexp_conversion_factor = data.attrs['aexp-scale-exponent']
                        cgs_conversion_factor = data.attrs['CGSConversionFactor']

                        if verbose:
                            print('Loading ',quantity)
                            print('h exponent = ',h_conversion_factor)
                            print('a exponent = ',aexp_conversion_factor)
                            print('CGS conversion factor = ',cgs_conversion_factor)

                        # Get the data type
                        dt = deepcopy(data.dtype)

                        data_arr = np.array(data,dtype=dt)
                
                    else:
                        data = f['/'+table+'/%s'%(quantity)]
                        data_arr = np.append(data_arr,np.array(data,dtype=dt),axis=0)

            except:
                # print('Run out of files at ',file_ind)
                break

            file_ind += 1
        
        if np.issubdtype(dt,np.integer):
            # Don't do any corrections if loading in integers
            return data_arr

        else:

            if phys_units:
                if cgs_units:
                    return data_arr * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor) * cgs_conversion_factor
                else:
                    return data_arr * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor)
            else:
                if cgs_units:
                    return data_arr * cgs_conversion_factor
                else:
                    return data_arr


    def fof(self,quantity,phys_units=True,cgs_units=False,verbose=False):

        return self.catalogue_read(quantity,table='FOF',phys_units=phys_units,cgs_units=cgs_units,verbose=verbose)

    def subfind(self,quantity,phys_units=True,cgs_units=False,verbose=False):

        return self.catalogue_read(quantity,table='Subhalo',phys_units=phys_units,cgs_units=cgs_units,verbose=verbose)



    def select(self,groupnumber,parttype=0,region_size='r200',region_shape='sphere'):
        '''
        Selection routine for CENTRALS only.
        Given a group number, selects a region of a given extent from that group's centre of potential.
        If region_shape is 'sphere' - select a spherical region of RADIUS region_size
        If region_shape is 'cube' - select a sub-box of SIDE LENGTH region_size
        Works with all particle types, default is 0 (gas)
        '''
        assert region_shape in ['sphere','cube'],'Please specify a valid region_shape (sphere or cube)'

        # Get the centre of potential from SUBFIND
        centre = self.central_COP[groupnumber-1,:]
        code_centre = centre * self.h/self.aexp # convert to h-less comoving code units

        # If the region size hasn't been given, set it to r200 (this is the default)
        if region_size == 'r200':

            region_size = self.r200[groupnumber-1]

            if region_shape == 'cube': # Double to give cube going out to r200
                region_size *= 2.

        code_region_size = region_size * self.h/self.aexp # convert to h-full comoving code units

        self.parttype = parttype
        self.this_M200 = self.M200[groupnumber-1]
        self.this_r200 = self.r200[groupnumber-1]
        self.this_groupnumber = groupnumber
        self.this_centre = centre
        self.region_size = region_size
        self.region_shape = region_shape

        # Open snapshot
        self.snap = self.read.EagleSnapshot(self.snapfile)

        # Select region of interest - this isolates the content of 'snap' to only this region for future use.
        self.snap.select_region(code_centre[0]-code_region_size,
                            code_centre[0]+code_region_size,
                            code_centre[1]-code_region_size,
                            code_centre[1]+code_region_size,
                            code_centre[2]-code_region_size,
                            code_centre[2]+code_region_size)

        # Now we just need to establish which of the particles we loaded in are within the spherical region.
        pos = self.snap.read_dataset(parttype,'Coordinates') * self.aexp/self.h

        if len(pos) == 0:
            # If no particles are found, return an empty array
            self.particle_selection = []
            self.have_run_select = True
            return

        # Wrap the box
        pos -= centre
        pos+=self.physical_boxsize/2.
        pos%=self.physical_boxsize
        pos-=self.physical_boxsize/2.

        # Create a mask to the region we want, for future use.

        if region_shape == 'sphere':
            r2 = np.einsum('...j,...j->...',pos,pos) # get the radii from the centre
            self.particle_selection = np.where(r2<region_size**2)[0] # make the mask   

        else:
            self.particle_selection = np.where((np.absolute(pos[:,0])<region_size/2.)&(np.absolute(pos[:,1])<region_size/2.)&(np.absolute(pos[:,2])<region_size/2.))[0] # make the mask  
        
        self.have_run_select = True

        


    def load(self,quantity,phys_units=True,cgs_units=False,verbose=False):
        '''
        Now we have run "select" and established which particles are in our spherical region, we can load
        in anything we want!
        '''

        # First make sure that select has been run
        assert self.have_run_select == True,'Please run "select" before trying to load anything in.'

        # Get our factors of h and a to convert to physical units
        pointto_snapfile = self.snapfile[:-6]
        snapfile_ind = 0
        while True:
            try:
                with h5.File(pointto_snapfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                    temp_data = f['/PartType%i/%s'%((self.parttype,quantity))]
                    h_conversion_factor = temp_data.attrs['h-scale-exponent']
                    aexp_conversion_factor = temp_data.attrs['aexp-scale-exponent']
                    cgs_conversion_factor = temp_data.attrs['CGSConversionFactor']

            except:
                print('No particles of type ',self.parttype,' in snapfile ',snapfile_ind)
                snapfile_ind += 1
                continue
            # print('Particles of type ',self.parttype,' FOUND in snapfile ',snapfile_ind)
            # print(h_conversion_factor, aexp_conversion_factor)
            break

        if verbose:
            print('Loading ',quantity)
            print('h exponent = ',h_conversion_factor)
            print('a exponent = ',aexp_conversion_factor)
            print('CGS conversion factor = ',cgs_conversion_factor)

        # Load in the quantity
        loaded_data = self.snap.read_dataset(self.parttype,quantity)

        dt = loaded_data.dtype

        # Cast the data into a numpy array of the correct type
        loaded_data = np.array(loaded_data,dtype=dt)

        if np.issubdtype(dt,np.integer):
            # Don't do any corrections if loading in integers
            return loaded_data[self.particle_selection]

        else:

            if phys_units:
                if cgs_units:
                    return loaded_data[self.particle_selection] * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor) * cgs_conversion_factor
                else:
                    return loaded_data[self.particle_selection] * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor)
            else:
                if cgs_units:
                    return loaded_data[self.particle_selection] * cgs_conversion_factor
                else:
                    return loaded_data[self.particle_selection]


    def get_Jvector(self,XYZ,aperture=0.03,CoMvelocity=True):

        Vxyz = self.load('Velocity')
        Jmass = self.load('Mass')

        particlesall = np.vstack([XYZ.T,Jmass,Vxyz.T]).T
        # Compute distances
        distancesall = np.linalg.norm(particlesall[:,:3],axis=1)
        # Restrict particles
        extract = (distancesall<aperture)
        particles = particlesall[extract].copy()
        # distances = distancesall[extract].copy()
        Mass = np.sum(particles[:,3])
        if CoMvelocity:
            # Compute CoM velocty & correct
            dvVmass = np.nan_to_num(np.sum(particles[:,3][:,np.newaxis]*particles[:,4:7],axis=0)/Mass)
            particlesall[:,4:7]-=dvVmass
            particles[:,4:7]-=dvVmass
        smomentums = np.cross(particles[:,:3],particles[:,4:7])
        momentum = np.sum(particles[:,3][:,np.newaxis]*smomentums,axis=0)
        Momentum = np.linalg.norm(momentum)
        # Compute cylindrical quantities
        zaxis = (momentum/Momentum)
        return zaxis


    def load_coordinates(self,align=None,alignaperture=0.01):
        '''
        Loads coordinates and wraps the box
        '''

        # First make sure that select has been run
        assert self.have_run_select == True,'Please run "select" before trying to load anything in.'

        assert align in ['face','edge',None],'Please pick a valid alignment'

        coords = self.load('Coordinates')

        # Centre and wrap the box
        coords -= self.this_centre
        coords+=self.physical_boxsize/2.
        coords%=self.physical_boxsize
        coords-=self.physical_boxsize/2.

        if align is not None:

            # Generate the new basis vectors
            J_vec = self.get_Jvector(coords,aperture=alignaperture)

            e3= J_vec #already normalised if using kinematics diagnostics

            e2 = np.ones(3)  # take a random vector
            e2 -= e2.dot(e3) * e3       # make it orthogonal to k
            e2 /= np.linalg.norm(e2)  # normalize it
            e1 = np.cross(e3,e2)
            e1 /= np.linalg.norm(e1)

            transformation_matrix = np.stack((e1.T,e2.T,e3.T), axis = 1 ).T

            for i in range(len(coords)):

                coords[i,:] = np.dot(transformation_matrix,coords[i,:])

            if align == 'face':

                return coords

            elif align == 'edge':
                
                coords_edge = np.empty(np.shape(coords))

                coords_edge[:,0] = coords[:,0]
                coords_edge[:,1] = coords[:,2]
                coords_edge[:,2] = coords[:,1]
                
                return coords_edge

        else:
            return coords


    def set_scene(self,quantity,camera_position=None,max_hsml=None,align=None,selection=None):
        '''
        Returns an instance of sphviewer.Scene, which can then be addressed for camera control, rendering etc.
        You can pass in a set of indices matching a selection you've already made, such that the loaded co-ordinates match.
        '''

        # First make sure that select has been run
        assert self.have_run_select == True,'Please run "select" before trying to load anything in.'

        if not 'sphviewer' in globals():
            sphviewer = import_module('sphviewer')

        # # Make sure that imaging is enabled
        # assert self.visualisation_enabled == True,'Please enable visualisation when initialising snapshot'

        pos = self.load_coordinates(align=align)

        if selection is not None:
            pos = pos[selection,:]
        else:
            selection = np.arange(len(pos[:,0]))

        assert len(pos[:,0])==len(quantity),'Size mismatch between input quantity and particle selection'

        if self.parttype in [1,2,3]:
            Particles = sphviewer.Particles(pos, quantity, hsml=None, nb=58)
        else:
            hsml = self.load('SmoothingLength')[selection]
            if max_hsml is not None:
                hsml[hsml>max_hsml] = max_hsml
            Particles = sphviewer.Particles(pos, quantity, hsml=hsml)

        Scene = sphviewer.Scene(Particles)

        if self.region_shape == 'cube':
            default_extent = self.region_size/2.
        else:
            default_extent = self.region_size

        if camera_position is not None:
            camera_position -= self.this_centre
        else:
            camera_position = [0.,0.,0.]

        Scene.update_camera(x=camera_position[0],y=camera_position[1],z=camera_position[2],r='infinity',extent=[-default_extent, default_extent, -default_extent, default_extent], xsize=1024, ysize=1024)

        return Scene


    ############################



    def load_abundances(self):

        '''
        Returns arrays of mass abundance and number ratio, as well as X_e, X_i and mu.
        '''

        # First make sure that select has been run
        assert self.have_run_select == True,'Please run "select" before trying to load anything in.'

        abunds = np.zeros((len(self.particle_selection),11))
        abunds[:,0] = self.load("SmoothedElementAbundance/Hydrogen")
        abunds[:,1] = self.load("SmoothedElementAbundance/Helium")
        abunds[:,2] = self.load("SmoothedElementAbundance/Carbon")
        abunds[:,3] = self.load("SmoothedElementAbundance/Nitrogen")
        abunds[:,4] = self.load("SmoothedElementAbundance/Oxygen")
        abunds[:,5] = self.load("SmoothedElementAbundance/Neon")
        abunds[:,6] = self.load("SmoothedElementAbundance/Magnesium")
        abunds[:,7] = self.load("SmoothedElementAbundance/Silicon")
        abunds[:,8] = abunds[:,7]*0.6054160
        abunds[:,9] = abunds[:,7]*0.0941736
        abunds[:,10] = self.load("SmoothedElementAbundance/Iron")


        masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
        atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
        Xe = np.ones(len(abunds[:,0])) # Initialise values for hydrogen
        Xi = np.ones(len(abunds[:,0]))
        mu = np.ones(len(abunds[:,0]))*0.5
        num_ratios = np.zeros(np.shape(abunds))
        for col in range(len(abunds[0,:])): # convert mX/mtot to mX/mH
            num_ratios[:,col] = abunds[:,col] / abunds[:,0]
        for element in range(len(abunds[0,:])-1):
            mu += num_ratios[:,element+1]/(1.+atomic_numbers[element+1])
            num_ratios[:,element+1] *= masses_in_u[0]/masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)
            Xe += num_ratios[:,element+1]*atomic_numbers[element+1] # Assuming complete ionisation
            Xi += num_ratios[:,element+1]

        return abunds, num_ratios, Xe, Xi, mu


    def attrs(self,quantity):
        '''
        Read the attributes of a given quantity in the snapshot and return a dictionary.
        This function is unrelated to 'select' - you must give the full quantity name i.e. /PartType0/Coordinates
        '''
        pointto_snapfile = self.snapfile[:-6]
        snapfile_ind = 0
        while True:
            try:
                with h5.File(pointto_snapfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                    temp_data = f[quantity]
                    return dict(temp_data.attrs)

            except:
                print('No particles of type ',self.parttype,' in snapfile ',snapfile_ind)
                snapfile_ind += 1
                continue
            break
        print('No particles found')
        exit()


    def attrs_subfind(self,quantity):
        '''
        Read the attributes of a given quantity in the subfind catalogues and return a dictionary.
        This function is unrelated to 'select' - you must give the full quantity name i.e. /PartType0/Coordinates
        '''
        snapfile_ind = 0
        while True:
            try:
                with h5.File(self.subfindfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                    temp_data = f[quantity]
                    return dict(temp_data.attrs)

            except:
                print('No haloes of type ',self.parttype,' in snapfile ',snapfile_ind)
                snapfile_ind += 1
                continue
            break
        print('No haloes found')
        exit()




def attrs(quantity,
                    box = 'L0100N1504',
                    model = 'REFERENCE',
                    tag='028_z000p000',
                    pdata_type = 'SNAPSHOT',
                    data_location = '/hpcdata0/simulations/EAGLE/'):
    '''
    Reads the attributes of 'quantity' and returns a dictionary of them.
    See what attributes are available with e.g. readattrs('Header').keys()
    Access them with e.g. readattrs('Header')['BoxSize']

    Arguments:

    quantity: The dataset you want the attributes of.
    box: The simulation volume, default is L0100N1504
    model: The physical model employed, default is REFERENCE
    tag: The snapshot id, default is 028_z000p000 (z=0)
    pdata_type: The type of snapshot, options are SNAPSHOT or PARTDATA, default is SNAPSHOT
    data_location: The parent directory for the EAGLE data at LJMU.
    '''

    if pdata_type == 'SNAPSHOT':
        snapfile = data_location + box + '/' + model + '/data/snapshot_'+tag+'/snap_'+tag+'.0.hdf5'
    elif pdata_type == 'PARTDATA':
        snapfile = data_location + box + '/' + model + '/data/particledata_'+tag+'/eagle_subfind_particles_'+tag+'.0.hdf5'
    elif pdata_type == 'SUBFIND':
        snapfile = data_location + box + '/' + model + '/data/groups_'+tag+'/eagle_subfind_tab_'+tag+'.0.hdf5'
    else:
        raise TypeError('Please pick a valid pdata_type (SNAPSHOT, PARTDATA or SUBFIND)')

    pointto_snapfile = snapfile[:-6]
    snapfile_ind = 0
    while True:
        try:
            with h5.File(pointto_snapfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                attributes = dict(f[quantity].attrs)
        except:
            snapfile_ind += 1
            continue
        break

    return attributes



def catalogue(filepath,groupnumbers,fields,gn_field='GroupNumber'):
    '''
    For loading in pre-computed catalogues by group number
    '''

    loaded_data = {}

    if isinstance(fields,str): # If only one field, put into a list to simplify code
        fields = [fields,]

    with h5.File(filepath, 'r') as f:
        file_groupnumbers = np.array(f[gn_field])                

        group_locations = np.zeros(len(groupnumbers),dtype=int)
        for i, g in enumerate(groupnumbers):
            try:
                group_locations[i] = np.where(file_groupnumbers==g)[0]
            except ValueError:
                raise ValueError('Group missing from file.')

        for field in fields:

            loaded_data[field] = np.array(f[field])[group_locations]

    return loaded_data

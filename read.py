# -*- coding: utf-8 -*-

import numpy as np

import h5py as h5
from sys import exit
from importlib import import_module
from astropy.cosmology import FlatLambdaCDM

class snapshot(object):

    def __init__(self,sim = 'L0100N1504',
                        run = 'REFERENCE',
                        tag='028_z000p000',
                        pdata_type = 'SNAPSHOT',
                        data_location = '/hpcdata0/simulations/EAGLE/',
                        GM = None,
                        seed = None,
                        visualisation=False):

        self.E = import_module('eagle')
        self.read = import_module('read_eagle')

        # if visualisation:
        #     self.sphviewer = import_module('sphviewer')
        #     self.visualisation_enabled = True
        # else:
        #     self.visualisation_enabled = False

        self.sim = sim
        self.run = run
        self.tag = tag
        self.pdata_type = pdata_type

        # # Extra path component for a genetically modified run
        # if GM is not None:
        #     data_location = '/hpcdata0/simulations/EAGLE/'
        #     gm_path_extra_1 = 'GM/'
        #     gm_path_extra_2 = '/'+GM+'/'+seed
        # else:
        #     gm_path_extra_1 = ''
        #     gm_path_extra_2 = ''

        # Name of one file from the snapshot
        if pdata_type == 'SNAPSHOT':
            self.snapfile = data_location + sim + '/' + run + '/data/snapshot_'+tag+'/snap_'+tag+'.0.hdf5'
        elif pdata_type == 'PARTDATA':
            self.snapfile = data_location + sim + '/' + run + '/data/particledata_'+tag+'/eagle_subfind_particles_'+tag+'.0.hdf5'
        else:
            raise TypeError('Please pick a valid pdata_type (SNAPSHOT or PARTDATA)')


        self.sim_path = data_location + sim + '/' + run + '/data/'

        # Get volume information
        boxsize = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/BoxSize")
        self.NumPart = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/NumPart_Total")
        self.h = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/HubbleParam")
        self.aexp = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/ExpansionFactor")
        self.z = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/Redshift")
        self.physical_boxsize = boxsize * self.aexp/self.h

        with h5.File(self.snapfile, 'r') as f:
            self.masstable = f['Header'].attrs['MassTable']/self.h

        self.cosmology = {}
        self.cosmology['Omega0'] = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/Omega0")
        self.cosmology['OmegaBaryon'] = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/OmegaBaryon")
        self.cosmology['OmegaLambda'] = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/OmegaLambda")
        self.cosmology['HubbleParam'] = self.E.readAttribute(self.pdata_type, self.sim_path, tag, "/Header/HubbleParam")

        self.f_b_cosmic = self.cosmology['OmegaBaryon']/self.cosmology['Omega0']

        # Get halo information
        first_subhalo = np.array(self.E.readArray("SUBFIND_GROUP", self.sim_path, self.tag, 'FOF/FirstSubhaloID'))

        # If the final FOF group is empty, it can be assigned a FirstSubhaloID which is out of range
        Mstar_30kpc = np.array(self.E.readArray("SUBFIND", self.sim_path, self.tag, "/Subhalo/ApertureMeasurements/Mass/030kpc"))
        max_subhalo = len(Mstar_30kpc)
        del Mstar_30kpc
        first_subhalo[first_subhalo==max_subhalo] -= 1 # this line fixes the issue

        self.subfind_all_gn = np.array(self.E.readArray('SUBFIND', self.sim_path, self.tag, 'Subhalo/GroupNumber'))
        self.subfind_all_sgn = np.array(self.E.readArray('SUBFIND', self.sim_path, self.tag, 'Subhalo/SubGroupNumber'))
        self.subfind_all_COP = np.array(self.E.readArray('SUBFIND', self.sim_path, self.tag, 'Subhalo/CentreOfPotential'))

        self.subfind_centres = np.array(self.E.readArray('SUBFIND', self.sim_path, self.tag, 'Subhalo/CentreOfPotential'))[first_subhalo, :]
        self.bulk_velocity = np.array(self.E.readArray("SUBFIND",self.sim_path,self.tag,'Subhalo/Velocity'))[first_subhalo,:]
        self.r200 = np.array(self.E.readArray("SUBFIND_GROUP", self.sim_path, self.tag, 'FOF/Group_R_Crit200'))
        self.M200 = np.array(self.E.readArray("SUBFIND_GROUP", self.sim_path, self.tag, 'FOF/Group_M_Crit200'))



        self.have_run_select = False # This is set to True when a region has been selected - prevents crashes later on.


    def astropy_cosmology(self):
        '''
        Create an astropy.cosmology object for doing cosmological calculations based on the cosmology in this snapshot.
        '''
        return FlatLambdaCDM(100.*self.cosmology['HubbleParam'],Om0=self.cosmology['Omega0'],Ob0=self.cosmology['OmegaBaryon'])
        


    def select(self,groupnumber,parttype=0,region_size='r200',region_shape='sphere'):
        '''
        Given a group number, selects a region of a given extent from that group's centre of potential.
        If region_shape is 'sphere' - select a spherical region of RADIUS region_size
        If region_shape is 'cube' - select a sub-box of SIDE LENGTH region_size
        Works with all particle types, default is 0 (gas)
        '''
        assert region_shape in ['sphere','cube'],'Please specify a valid region_shape (sphere or cube)'

        # Get the centre of potential from SUBFIND
        centre = self.subfind_centres[groupnumber-1,:]
        code_centre = centre * self.h/self.aexp # convert to h-less comoving code units

        # If the region size hasn't been given, set it to r200 (this is the default)
        if region_size == 'r200':

            region_size = self.r200[groupnumber-1]

            if region_shape == 'cube': # Double to give cube going out to r200
                region_size *= 2.

        code_region_size = region_size * self.h/self.aexp # convert to h-full comoving code units

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


        self.parttype = parttype
        self.have_run_select = True

        self.this_M200 = self.M200[groupnumber-1]
        self.this_r200 = self.r200[groupnumber-1]
        self.this_groupnumber = groupnumber
        self.this_centre = centre
        self.region_size = region_size
        self.region_shape = region_shape


    def load(self,quantity,verbose=False):
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

            except:
                print 'No particles of type ',self.parttype,' in snapfile ',snapfile_ind
                snapfile_ind += 1
                continue
            # print 'Particles of type ',self.parttype,' FOUND in snapfile ',snapfile_ind
            # print h_conversion_factor, aexp_conversion_factor
            break

        if verbose:
            print 'Loading ',quantity
            print 'h exponent = ',h_conversion_factor
            print 'a exponent = ',aexp_conversion_factor

        # Load in the quantity
        loaded_data = self.snap.read_dataset(self.parttype,quantity)

        return loaded_data[self.particle_selection] * np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor)




    def get_Jvector(self,XYZ,aperture=0.03,CoMvelocity=True):

        Vxyz = self.load('Velocity')
        Jmass = self.load('Mass')

        particlesall = np.vstack([XYZ.T,Jmass,Vxyz.T]).T
        # Compute distances
        distancesall = np.linalg.norm(particlesall[:,:3],axis=1)
        # Restrict particles
        extract = (distancesall<aperture)
        particles = particlesall[extract].copy()
        distances = distancesall[extract].copy()
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

        if not sphviewer in sys.modules:
            self.sphviewer = import_module('sphviewer')

        # # Make sure that imaging is enabled
        # assert self.visualisation_enabled == True,'Please enable visualisation when initialising snapshot'

        pos = self.load_coordinates(align=align)

        if selection is not None:
            pos = pos[selection,:]
        else:
            selection = np.arange(len(pos[:,0]))

        assert len(pos[:,0])==len(quantity),'Size mismatch between input quantity and particle selection'

        if self.parttype in [1,2,3]:
            Particles = self.sphviewer.Particles(pos, quantity, hsml=None, nb=58)
        else:
            hsml = self.load('SmoothingLength')[selection]
            if max_hsml is not None:
                hsml[hsml>max_hsml] = max_hsml
            Particles = self.sphviewer.Particles(pos, quantity, hsml=hsml)

        Scene = self.sphviewer.Scene(Particles)

        if self.region_shape == 'cube':
            default_extent = self.region_size/2.
        else:
            default_extent = self.region_size

        Scene.update_camera(x=0.,y=0.,z=0.,r='infinity',extent=[-default_extent, default_extent, -default_extent, default_extent], xsize=1024, ysize=1024)

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
    For loading in catalogues by group number
    '''

    loaded_data = {}

    if isinstance(fields,basestring): # If only one field, put into a list to simplify code
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
# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5
from sys import exit
from importlib import import_module
from astropy.cosmology import FlatLambdaCDM
from copy import deepcopy
from warnings import warn

from pyread_eagle import EagleSnapshot


class MaskedReadEagleSnapshot(EagleSnapshot):
    """
    An instance of pyread_eagle.EagleSnapshot that is already masked to a certain region.
    These make it possible for users to just have one instance of eagle_tools.Snapshot in their code, which generates these 
    masked EagleSnapshots for a given region geometry and particle type.
    The old method would mask the eagle_tools.Snapshot object internally which was more cumbersome than I expected,
    particularly when working with different particle types at the same time.

    This class shouldn't really be used on its own, but rather passed back into eagle_tools.Snapshot.load().

    NB this class works entirely in CODE units, as pyread_eagle does. Users should be careful that all inputs, outputs and 
    properties are treated as such.

    """

    def __init__(
        self,
        filepath,
        parttype,
        centre,
        shape,
        boxsize,
        radius = None,
        side_length = None
    ):
        """
        NB: all inputs must be in CODE units
        """

        self.parttype = parttype
        self.centre = centre

        if shape == 'sphere':
            if side_length is not None:
                print("Argument `side_length` specified, but shape is `sphere`. `side_length` will be ignored in favour of `radius`.")
            if radius is None:
                raise ValueError("Please specify the `radius` kwarg for a spherical selection.")
            region_size = radius

        elif shape == 'cube':
            if radius is not None:
                print("Argument `radius` specified, but shape is `cube`. `radius` will be ignored in favour of `side_length`.")
            if side_length is None:
                raise ValueError("Please specify the `radius` kwarg for a spherical selection.")
            region_size = side_length/2.

        else:
            raise ValueError("Selected shape must be one of `sphere` or `cube`.")
        
        super().__init__(filepath)

        self.dataset_names = self.dataset_names[parttype]

        # Select region of interest
        super().select_region(centre[0]-region_size,
                            centre[0]+region_size,
                            centre[1]-region_size,
                            centre[1]+region_size,
                            centre[2]-region_size,
                            centre[2]+region_size)

        # Now we just need to establish which of the particles we loaded in are within the spherical region.
        loaded_pos = super().read_dataset(parttype,'Coordinates')

        if len(loaded_pos) == 0:
            # If no particles are found, return an empty array
            self.mask = []
            warn(f"No particles found...", RuntimeWarning)

        else:

            # Wrap the box
            pos = loaded_pos - centre
            pos += boxsize/2.
            pos %= boxsize
            pos -= boxsize/2.

            # Create a mask to the region we want, for future use.
            if shape == 'sphere':
                r2 = np.einsum('...j,...j->...',pos,pos)
                self.mask = np.where(r2<radius**2)[0] 
            else:
                self.mask = np.where((np.absolute(pos[:,0])<side_length/2.)&(np.absolute(pos[:,1])<side_length/2.)&(np.absolute(pos[:,2])<side_length/2.))[0]
        
    def select_region(self, *args, **kwargs):
        raise SyntaxError(f"`select_region` has already been run on initialisation. Running again will break the internal mask. Instead, users should initialise a new MaskedReadEagleSnapshot.")

    def read_dataset(self, *args, **kwargs):

        return super().read_dataset(self.parttype, *args, **kwargs)[self.mask]



class Snapshot(object):

    def __init__(self,
        sim = 'L0100N1504',
        model = 'REFERENCE',
        tag='028_z000p000',
        pdata_type = 'SNAPSHOT',
        data_location = '/hpcdata0/simulations/EAGLE/'
    ):

        self.sim = sim
        self.model = model
        self.tag = tag
        self.pdata_type = pdata_type
        self.sim_path = f"{data_location}/{sim}/{model}/data"

        # Create strings to point the read module to the first snapshot/particle/subfind files
        if pdata_type == 'SNAPSHOT':
            # self.snapfile = self.sim_path + 'snapshot_'+tag+'/snap_'+tag+'.0.hdf5'
            self.snapfile = f"{self.sim_path}/snapshot_{tag}/snap_{tag}.0.hdf5"
        elif pdata_type == 'PARTDATA':
            # self.snapfile = self.sim_path + 'particledata_'+tag+'/eagle_subfind_particles_'+tag+'.0.hdf5'
            self.snapfile = f"{self.sim_path}/particledata_{tag}/eagle_subfind_particles_{tag}.0.hdf5"
        else:
            raise TypeError('Please pick a valid pdata_type (SNAPSHOT or PARTDATA)')
        # self. = self.sim_path + 'groups_'+tag+'/eagle_subfind_tab_' + tag + '.'
        self.subfind_root = f"{self.sim_path}/groups_{tag}/eagle_subfind_tab_{tag}"

        # Get volume information. Note that header is taken from one file only, so quantities such as 'NumPart_ThisFile' are not useful.
        self.header = self.attrs('Header')

        # Assign some of the most useful quantities to the snapshot object
        self.NumPart = self.header['NumPart_Total']
        self.h = self.header['HubbleParam']
        self.aexp = self.header['ExpansionFactor']
        self.z = self.header['Redshift']
        self.masstable = self.header['MassTable']/self.h
        self.code_boxsize = self.header['BoxSize']
        self.physical_boxsize = self.code_boxsize * self.aexp/self.h
        self.f_b_cosmic = self.header['OmegaBaryon']/self.header['Omega0']

        # Create an astropy.cosmology object for doing cosmological calculations.
        self.cosmology = FlatLambdaCDM(100.*self.header['HubbleParam'],Om0=self.header['Omega0'],Ob0=self.header['OmegaBaryon'])

        # Fields in this dictionary should always have physical units.
        #TODO enforce this with a custom setter
        self.subfind = {}
        self.subfind_attrs = {}


    def select_halo(self,
        groupnumber: int,
        subgroupnumber: int,
        parttype: int = None,
        shape: str = 'sphere',
        radius: str | float = None,
        side_length: str | float = None
    ) -> MaskedReadEagleSnapshot:
        
        if parttype is None:
            raise ValueError("Please enter a `parttype` as an integer kwarg.")
        
        if shape == 'sphere':
            if side_length is not None:
                print("Argument `side_length` specified, but shape is `sphere`. `side_length` will be ignored in favour of `radius`.")
            if radius is None:
                raise ValueError("Please specify the `radius` kwarg for a spherical selection.")
            region_size = radius

        elif shape == 'cube':
            if radius is not None:
                print("Argument `radius` specified, but shape is `cube`. `radius` will be ignored in favour of `side_length`.")
            if side_length is None:
                raise ValueError("Please specify the `radius` kwarg for a spherical selection.")
            region_size = side_length

        else:
            raise ValueError("Selected shape must be one of `sphere` or `cube`.")

        if not 'Subhalo/CentreOfPotential' in self.subfind.keys():
            self.load_subfind('Subhalo/CentreOfPotential')

        if not 'Subhalo/GroupNumber' in self.subfind.keys() and 'Subhalo/SubGroupNumber' in self.subfind.keys():
            self.load_subfind('Subhalo/GroupNumber','Subhalo/SubGroupNumber')

        subfind_location = np.where(
            (self.subfind['Subhalo/GroupNumber'] == groupnumber)&
            (self.subfind['Subhalo/SubGroupNumber'] == subgroupnumber)
        )[0][0]

        # Make sure we're working with a copy as this has caused issues before
        # The internally cached subfind quantity must be in physical units
        centre = deepcopy(
            self.subfind['Subhalo/CentreOfPotential'][subfind_location]
        ) * self.h/self.aexp # convert back to code units

        if isinstance(region_size,str):
            if subgroupnumber != 0 and region_size[:3] == 'FOF':
                raise ValueError(f"Requested a region size from the FOF table but halo is not a central (SubGroupNumber=0).")
            elif subgroupnumber == 0 and region_size[:3] == 'FOF':
                region_size_location = groupnumber - 1
            else:
                region_size_location = subfind_location
                
            # Look to see if the relevant subfind table has already been cached
            if not region_size in self.subfind.keys():
                self.load_subfind(region_size)
            # Make sure we're working with a copy as this has caused issues before
            # The internally cached subfind quantity must be in physical units
            region_size_to_load = deepcopy(
                self.subfind[region_size][region_size_location]
            ) * self.h/self.aexp # convert back to code units
        
        else:
            region_size_to_load = region_size * self.h/self.aexp # assuming input was in physical units

        return MaskedReadEagleSnapshot(
            self.snapfile,
            parttype,
            centre,
            shape,
            self.code_boxsize,
            radius = region_size_to_load if shape == 'sphere' else None,
            side_length = region_size_to_load if shape == 'cube' else None
        )

    
    def load(self,
        masked_snapshot: MaskedReadEagleSnapshot,
        quantity: str,
        phys_units: bool = True,
        cgs_units: bool = False,
        verbose: bool = False,
        centre_coords: bool = True,
        wrap_coords: bool = True,
        align_coords: bool = None,
        align_coords_aperture: float = 0.01
    ):

        assert isinstance(masked_snapshot, MaskedReadEagleSnapshot)

        # Get our factors of h and a to convert to physical units
        pointto_snapfile = self.snapfile[:-6]
        snapfile_ind = 0
        while True:
            try:
                with h5.File(pointto_snapfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                    temp_data = f['/PartType%i/%s'%((masked_snapshot.parttype,quantity))]
                    h_conversion_factor = temp_data.attrs['h-scale-exponent']
                    aexp_conversion_factor = temp_data.attrs['aexp-scale-exponent']
                    cgs_conversion_factor = temp_data.attrs['CGSConversionFactor']
            except:
                print('No particles of type ',masked_snapshot.parttype,' in snapfile ',snapfile_ind)
                snapfile_ind += 1
                continue
            break

        if verbose:
            print('Loading ',quantity)
            print('h exponent = ',h_conversion_factor)
            print('a exponent = ',aexp_conversion_factor)
            print('CGS conversion factor = ',cgs_conversion_factor)

        # Load in the quantity
        loaded_data = masked_snapshot.read_dataset(quantity)

        # Cast the data into a numpy array of the correct type
        dt = loaded_data.dtype
        loaded_data = np.array(loaded_data,dtype=dt)

        if not np.issubdtype(dt,np.integer): # Don't do any corrections if loading in integers
            
            if phys_units:
                loaded_data *= np.power(self.h,h_conversion_factor) * np.power(self.aexp,aexp_conversion_factor)

            if cgs_units:
                loaded_data *= cgs_conversion_factor

        if quantity == 'Coordinates':
            loaded_data = self._transform_coordinates(
                                                    loaded_data,
                                                    masked_snapshot.centre * self.aexp/self.h,
                                                    centre_coords = centre_coords,
                                                    wrap_coords = wrap_coords,
                                                    align_coords = align_coords,
                                                    align_coords_aperture = align_coords_aperture
                                                    )
            
        return loaded_data


    def load_subfind(self,*quantities,verbose=False,overwrite=False):
        """
        Load subfind catalogues internally.
        We always internally cache in physical code units.
        cgs, aexp and h conversion factors can be found in Snapshot.subfind_attrs.
        """

        for quantity in quantities:

            if quantity in self.subfind.keys() and overwrite == False:
                print(f"{quantity} already cached.",flush=True)
                continue

            file_ind = 0
            while True: #TODO replace this implicit True with iterating over globbed files or something

                try:
                    with h5.File(f"{self.subfind_root}.{file_ind}.hdf5", 'r') as f:
                        
                        data = f[f"/{quantity}"]

                        if file_ind == 0:

                            # Grab conversion factors from first file
                            attrs = dict(data.attrs)

                            if verbose:
                                print('Loading ',quantity)
                                print('h exponent = ',attrs['h-scale-exponent'])
                                print('a exponent = ',attrs['aexp-scale-exponent'])
                                print('CGS conversion factor = ',attrs['cgs_conversion_factor'])

                            # Get the data type
                            dt = deepcopy(data.dtype)
                            loaded_data = np.array(data,dtype=dt)
                    
                        else:
                            loaded_data = np.append(loaded_data,np.array(data,dtype=dt),axis=0)

                except:
                    break

                file_ind += 1
            
            if not np.issubdtype(dt,np.integer): # Don't do any corrections if loading in integers

                loaded_data *= np.power(self.h,attrs['h-scale-exponent'])\
                            * np.power(self.aexp,attrs['aexp-scale-exponent'])\

            self.subfind[quantity] = loaded_data
            self.subfind_attrs[quantity] = attrs


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

    def softening(self, comoving = False):

        # This sucks, there must be a smarter way of getting the resolution
        if self.sim in ['L0025N0376', 'L0050N0752', 'L0100N1504']:
            e_com = 2.66
            e_max_phys = 0.7
        elif self.sim in ['L0025N0752', 'L0034N1034']:
            e_com = 1.33
            e_max_phys = 0.35
        else:
            raise NameError('Softening for this simulation not known yet!')
    
        # Logic for the comoving softening
        if e_com < e_max_phys/self.aexp:
            e = e_com
        else:
            e = e_max_phys/self.aexp
        
        return e if comoving else e*self.aexp

    def _get_Jvector(self,XYZ,aperture=0.03,CoMvelocity=True):

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


    def _transform_coordinates(self,
            coords,
            centre,
            centre_coords = True,
            wrap_coords = True,
            align_coords=None,
            align_coords_aperture=0.01
    ):
        '''
        Wraps coordinates and optionally transforms them to align face-on or edge-on.
        '''

        if centre_coords:
            coords -= centre
            if wrap_coords:                
                coords+=self.physical_boxsize/2.
                coords%=self.physical_boxsize
                coords-=self.physical_boxsize/2.

        if align_coords is not None:

            assert align_coords in ['face','edge',None],'Please pick a valid alignment'

            # Generate the new basis vectors
            J_vec = self._get_Jvector(coords,aperture=align_coords_aperture)

            e3= J_vec #already normalised if using kinematics diagnostics

            e2 = np.ones(3)  # take a random vector
            e2 -= e2.dot(e3) * e3       # make it orthogonal to k
            e2 /= np.linalg.norm(e2)  # normalize it
            e1 = np.cross(e3,e2)
            e1 /= np.linalg.norm(e1)

            transformation_matrix = np.stack((e1.T,e2.T,e3.T), axis = 1 ).T

            #JD: this really sucks, I'm sure this can be vectorised
            for i in range(len(coords)):

                coords[i,:] = np.dot(transformation_matrix,coords[i,:])

            if align_coords == 'face':

                return coords

            # This also sucks
            elif align_coords == 'edge':
                
                coords_edge = np.empty(np.shape(coords))

                coords_edge[:,0] = coords[:,0]
                coords_edge[:,1] = coords[:,2]
                coords_edge[:,2] = coords[:,1]
                
                return coords_edge

        else:
            return coords




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

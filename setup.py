from setuptools import setup

setup(name = 'eagle_tools',
      author = 'Jon Davies',
      author_email = 'j.j.davies@ljmu.ac.uk',
      version = '0.1',
      description = 'Useful tools for working with EAGLE snapshots in python.',
      packages = ['eagle_tools',],
      install_requires=['numpy','h5py','astropy','pyread_eagle','eaglesqlTools',],
      )
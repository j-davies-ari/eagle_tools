# -*- coding: utf-8 -*-

import numpy as np
import os as os_module
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import argv, path, exit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr



class canvas(object):

    def __init__(self,figsize=(8,10),hspace=0.,wspace=0.,save_directory='/home/arijdav1/Dropbox/phd/figures/'):
        
        if not os_module.path.exists(save_directory):
            os_module.makedirs(save_directory)
        
        self.save_directory = save_directory

        self.default_axis = None

        self.fig = plt.figure(figsize=figsize)
        self.grid = plt.GridSpec(figsize[1], figsize[0], hspace=hspace, wspace=wspace)

    def add_axis(self,span,set_default=False,sharex=None,sharey=None):

        added_axis = self.fig.add_subplot(span,sharex=sharex,sharey=sharey)

        if set_default:
            self.default_axis = added_axis

        return added_axis


    def set_axis(self,axis):
        if axis != None:
            return axis
        else:
            if self.default_axis is not None:
                return self.default_axis
            else:
                raise ValueError('Please specify an axis or set one as default')

    def add_inset(self,location,axis=None,width="25%",height="35%",borderpad=2):
        a = self.set_axis(axis)

        if location == 'upper right':
            loc = 1
        elif location == 'lower right':
            loc = 4
        elif location == 'lower left':
            loc = 3
        elif location == 'upper left':
            loc = 2
        elif location in [1,2,3,4]:
            loc = location
        else:
            print 'Please specify a valid inset axis location'
            exit()

        return inset_axes(a, width=height, height=height, loc=loc,borderpad=borderpad)

    def line(self,xvals,yvals,axis=None,c='k',lw=2,ls='-',label=None):
        a = self.set_axis(axis)
        a.plot(xvals,yvals,c=c,lw=lw,ls=ls,label=label)

    def shade(self,xvals,y0,y1,axis=None,c='k',alpha=0.5):
        a = self.set_axis(axis)
        a.fill_between(xvals,y0,y1,color=c,alpha=alpha)

    def scatter(self,xvals,yvals,axis=None,c='k',s=10,marker='o',label=None):
        a = self.set_axis(axis)
        a.scatter(xvals,yvals,c=c,s=s,lw=0,marker=marker,label=label)

    def xlabel(self,label,axis=None,fontsize=22,loc='bottom'):
        assert loc in ['top','bottom'],'Please specify a valid location (bottom or top)'
        a = self.set_axis(axis)
        a.set_xlabel(label,fontsize=fontsize)
        if loc == 'top':
            a.xaxis.set_label_position(loc)
            a.xaxis.tick_top()
            a.xaxis.set_ticks_position('both')

    def ylabel(self,label,axis=None,fontsize=22,loc='left'):
        assert loc in ['left','right'],'Please specify a valid location (left or right)'
        a = self.set_axis(axis)
        a.set_ylabel(label,fontsize=fontsize)
        if loc == 'right':
            a.yaxis.set_label_position(loc)
            a.yaxis.tick_right()
            a.yaxis.set_ticks_position('both')


    def xlim(self,xmin,xmax,axis=None):
        a = self.set_axis(axis)
        a.set_xlim(xmin,xmax)

    def ylim(self,ymin,ymax,axis=None):
        a = self.set_axis(axis)
        a.set_ylim(ymin,ymax)


    def legend(self,axis=None,loc='lower right',ncol=1,size=15):

        a = self.set_axis(axis)

        if loc == 'above':
            a.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=ncol)

        elif loc == 'right_out':
            box = a.get_position()
            a.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            a.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True,prop={'size': size})

        else:
            a.legend(loc=loc,fancybox=True,prop={'size': size})

    def annotate(self,annotation,location,axis=None,fontsize=16):
        a = self.set_axis(axis)
        a.annotate(annotation,xy=location,fontsize=fontsize)

    def hide_xticks(self,axis=None):
        a = self.set_axis(axis)
        a.set_xticklabels([])

    def hide_yticks(self,axis=None):
        a = self.set_axis(axis)
        a.set_yticklabels([])

    def set_ticksize(self,labelsize,axis=None):
        a = self.set_axis(axis)
        a.tick_params(axis='both',labelsize=labelsize)

    def save(self,name,dpi=200,rasterised=True,tight_layout=True):
        if tight_layout:
            plt.tight_layout()
        self.fig.savefig(self.save_directory+name+'.pdf',dpi=dpi,rasterised=rasterised)

    def show(self):
        plt.show(self.fig)

    def close(self):
        plt.close(self.fig)



def get_bincentres(bin_edges):

    return (bin_edges[1:] + bin_edges[:-1])/2.


def NFW_profile(r_over_r200,c):

    return 1./(3. * (np.log(1+c)-c/(1.+c)) * r_over_r200 * ((1./c)+r_over_r200)**2)



def get_moving_spearman_rank(xs,ys,colours,
                            window_sizes=[300,100],
                            window_steps=[50,25],
                            transition_points=[12.,]):
    '''
    This function computes the moving Spearman rank correlation coefficient in a 3-dimensional dataset.
    The correlation between two quantities ('ys' and 'colours') are computed at fixed 'xs', which MUST be in ascending order.
    The user must set the size of the window in xs ('window_sizes') in which to compute the correlation between ys and colours
    and the 'step' through which this window moves in xs (window_steps). This window can vary over the range of xs,
    in which case the user must specify 'transition points' at which the window sizes change.

    The names 'xs', 'ys' and 'colours' refer to the elements on a plot which would typically be made using this function.
    For examples, see all plots in Davies, J. J. et al. (2019), MNRAS.
    The default values correspond to the parameters used in this paper. (xs = M200 in EAGLE)

    Arguments:

    xs - the values which we want to keep fixed while computing the Spearman rank correlation coefficient. Must be in ascending order.
    ys, colours - the values we want to compute the correlation between
    window_sizes - the number of datapoints in the moving window. This can have multiple values if you want the size of the window to change
                    at certain points in xs. This can be an int or a list.
    window_steps - the number of datapoints to step through when the window moves. If a list, must have the same length as window_sizes.
    transition_points - the values of xs at which to change the size/step of the window. Must have length len(window_sizes) - 1.


    Returns:

    centres - the centres of the moving window as it moved through xs.
    moving_rank - the moving Spearman rank coefficients
    moving_pvalue - the moving p-values, indicating the significance of the correlation in the moving window.

    '''

    assert all(xs[i] <= xs[i+1] for i in xrange(len(xs)-1)), "xs MUST be in ascending order."

    if isinstance(window_sizes,list):

        assert len(window_steps) == len(window_sizes), "window_sizes and window_steps must be of equal length"
        assert len(transition_points) == len(window_sizes)-1, "The number of transition points must be 1 less than the number of window sizes"

        # Make the moving window by defining 'starts' and 'stops'.

        starts = np.arange(len(xs[xs<transition_points[0]])-(window_sizes[0]/2)-1,step=window_steps[0])
        stops = starts + window_sizes[0]
        centres = starts + window_sizes[0]/2

        for i in range(len(window_sizes)): # Need to 

            if i == 0:
                continue

            if i == len(window_sizes)-1:
                starts_chunk = np.arange(len(xs[xs<transition_points[i-1]])-(window_sizes[i]/2),len(xs)-window_sizes[i],step=window_steps[i])
            else:
                starts_chunk = np.arange(len(xs[xs<transition_points[i-1]])-(window_sizes[i]/2),len(xs[xs<transition_points[i]])-(window_sizes[i]/2)-1,step=window_steps[i])

            starts = np.hstack((starts,starts_chunk))
            stops = np.hstack((stops,starts_chunk+window_sizes[i]))
            centres = np.hstack((centres,starts_chunk+window_sizes[i]/2))

    elif isinstance(window_sizes,int):

        assert isinstance(window_steps,int), 'Multiple window steps given for one window size.'

        # Make the moving window by defining 'starts' and 'stops'.

        starts = np.arange(len(xs)-window_sizes,step=window_steps)
        stops = starts + window_sizes
        centres = starts + window_sizes/2

    else:
        raise ValueError('Please give a valid window size(s) (int or list of ints)')


    # Initialise arrays to hold the moving ranks, and their associated p-values
    moving_rank = np.zeros(len(starts))
    moving_pvalue = np.zeros(len(starts))

    # Step through the bins
    for r in range(len(moving_rank)):
    	# Get the correlation coefficient and p value, which tells you how well ys and colours are correlated in each bin
        moving_rank[r], moving_pvalue[r] = spearmanr(ys[starts[r]:stops[r]],colours[starts[r]:stops[r]])

    return centres, moving_rank, moving_pvalue

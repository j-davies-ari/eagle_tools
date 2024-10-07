# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import statsmodels.nonparametric.smoothers_lowess as lowess
from dataclasses import dataclass


class config(object): # Sets the font of axis labels in your code. You can also use plotparams.columnwidth etc for sizes.
    def __init__(self):
        params = {"text.usetex": True, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'lines.linewidth' : 2, 'axes.titlesize' : 16, 'font.family' : 'serif'}
        plt.rcParams.update(params)
        
        self.columnwidth = 240./72.27
        self.textwidth = 504.0/72.27
        self.textheight = 682.0/72.27


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

    assert all(xs[i] <= xs[i+1] for i in range(len(xs)-1)), "xs MUST be in ascending order."

    if isinstance(window_sizes,list):

        assert len(window_steps) == len(window_sizes), "window_sizes and window_steps must be of equal length"
        assert len(transition_points) == len(window_sizes)-1, "The number of transition points must be 1 less than the number of window sizes"

        # Make the moving window by defining 'starts' and 'stops'.

        starts = np.arange(len(xs[xs<transition_points[0]])-(window_sizes[0]//2)-1,step=window_steps[0])
        stops = starts + window_sizes[0]
        centres = starts + window_sizes[0]//2

        for i in range(len(window_sizes)): 

            if i == 0:
                continue

            if i == len(window_sizes)-1:
                starts_chunk = np.arange(len(xs[xs<transition_points[i-1]])-(window_sizes[i]//2),len(xs)-window_sizes[i],step=window_steps[i])
            else:
                starts_chunk = np.arange(len(xs[xs<transition_points[i-1]])-(window_sizes[i]//2),len(xs[xs<transition_points[i]])-(window_sizes[i]//2)-1,step=window_steps[i])

            starts = np.hstack((starts,starts_chunk))
            stops = np.hstack((stops,starts_chunk+window_sizes[i]))
            centres = np.hstack((centres,starts_chunk+window_sizes[i]//2))

    elif isinstance(window_sizes,int):

        assert isinstance(window_steps,int), 'Multiple window steps given for one window size.'

        # Make the moving window by defining 'starts' and 'stops'.

        starts = np.arange(len(xs)-window_sizes,step=window_steps)
        stops = starts + window_sizes
        centres = starts + window_sizes//2

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



###########

# Helper functions

def _median_and_delta(xs,ys,lowess_frac=0.2):
    running_median = lowess.lowess(ys, xs, frac=lowess_frac, it=3, delta=0.0, is_sorted=True, missing='drop', return_sorted=False)
    return running_median, ys-running_median


@dataclass
class quantity:
    name: str
    data: np.array
    limits: tuple = (None,None)
    delta_limits: tuple = (None,None)
    label: str = None


# Plotting functions

def coloured_scatter(
    x: quantity,
    y: quantity,
    c: quantity,
    ax,
    method = 'hexbin',
    delta_y = False,
    delta_c = False,
    cmap='turbo',
    s=10,
    lw=0.3,
    hexbin_nx = 100,
    hexbin_ny = 30,
    hexbin_reduce = np.mean,
    show_xlabels=True,
    show_ylabels=True,
    label_size=22,
    lowess_frac = 0.2,
    lowess_trim=10,
    rho_ax=None,
    rho_window_sizes=[300,100],
    rho_window_steps=[50,25],
    rho_transition_points=[12.,],
    return_savestring = True
):
    '''
    Generate a standard plot combo of a coloured scatter and moving rho on a given ax and rho_ax.
    Computes the moving rank based on default settings for plots vs. M200 - can define other settings as kwargs.
    Can also set how many datapoints to trim from each end of the running median line.
    '''

    assert method in ['hexbin','scatter'],"Argument `method` must be either `hexbin` or `scatter`."

    for input_quantity in [x,y,c]:
        if not isinstance(input_quantity,quantity):
            raise TypeError("Input variables must be eagle_tools.plot.quantity objects.")

    xdata = x.data
    ydata = y.data
    cdata = c.data

    # Check if sorted
    if not np.all(xdata[:-1] <= xdata[1:]):
        sort = np.argsort(xdata)
        xdata = xdata[sort]
        ydata = ydata[sort]
        cdata = cdata[sort]

    trim = lambda x: x[lowess_trim:-lowess_trim]

    # Find the median y and c as a function of x, and the residuals
    ymed, ydel = _median_and_delta(xdata,ydata,lowess_frac)
    cmed, cdel = _median_and_delta(xdata,cdata,lowess_frac)

    ydata = ydel if delta_y else ydata
    cdata = cdel if delta_c else cdata
    clims = c.delta_limits if delta_c else c.limits

    # Make the scatter plot.

    if method == 'hexbin':
        scatter = ax.hexbin(xdata,ydata,C=cdata,reduce_C_function=hexbin_reduce,cmap=cmap,vmin=clims[0],vmax=clims[1],gridsize=(hexbin_nx,hexbin_ny),lw=lw,edgecolor='gray')
    elif method == 'scatter':
        scatter = ax.scatter(xdata,ydata,c=cdata,cmap=cmap,vmin=clims[0],vmax=clims[1],s=s,lw=lw,edgecolor='gray',rasterized=True)
    else:
        # we should never get here
        raise RuntimeError

    ax.plot(trim(xdata),trim(ymed),c='w',lw=4)
    ax.plot(trim(xdata),trim(ymed),c='k',lw=2)

    # Set axis limits
    ax.set_ylim(y.limits)
    ax.set_xlim(x.limits)

    if show_ylabels:
        ax.set_ylabel(y.label,fontsize=22)
    else:
        ax.set_yticklabels([])

    # Moving spearman rank axis
    if rho_ax is not None:
        # Hide the x labels from the main axis
        # ax.set_xticklabels([])

        # Compute the moving spearman rank, return the window centres, ranks and p-values.
        rho_centres, rho, p = get_moving_spearman_rank(xdata, ydata, cdata, window_sizes=rho_window_sizes, window_steps=rho_window_steps, transition_points=rho_transition_points)

        # Plot the running ranks
        rho_ax.axhline(0,c='gray',lw=1)
        rho_ax.plot(xdata[rho_centres],rho,lw=2,c='k')

        rho_ax.set_xlim(x.limits)
        rho_ax.set_ylim(-1.,1.)

        if not show_xlabels:
            rho_ax.set_xticklabels([])
        else:
            rho_ax.set_xlabel(x.label,fontsize=label_size)

        if not show_ylabels:
            rho_ax.set_yticklabels([])
        else:
            rho_ax.set_ylabel(r'$\rho$',fontsize=label_size)

    else:
        # If no spearman rank axis, show the x label on the primary axis
        if show_xlabels:
            ax.set_xlabel(x.label,fontsize=label_size)
        else:
            pass

    # Return the mappable scatter for plotting colourbars.

    if return_savestring:
        return scatter, f"{x.name}_{"delta_" if delta_y else ""}{y.name}_{"delta_" if delta_c else ""}{c.name}"
    else:
        return scatter


def add_colourbar(mappable,cbar_axis,
                    label='',
                    location='top',
                    label_size=22):

    assert location in ['top','bottom','left','right'],'Colourbar location must be top, bottom, left or right'

    if location == 'top':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'horizontal', ticklocation = 'top')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'bottom':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'horizontal', ticklocation = 'bottom')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'right':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'vertical', ticklocation = 'right')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'left':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'vertical', ticklocation = 'left')
        cbar.set_label(label, labelpad=10,fontsize=label_size)


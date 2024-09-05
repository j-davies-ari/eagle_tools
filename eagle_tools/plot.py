# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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

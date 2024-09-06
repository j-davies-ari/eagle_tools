import numpy as np
import os as os_module
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class canvas(object):

    def __init__(self,figsize=(8,10),hspace=0.,wspace=0.,save_directory='/home/arijdav1/Dropbox/phd/figures/'):
        
        if not os_module.path.exists(save_directory):
            os_module.makedirs(save_directory)
        
        self.save_directory = save_directory

        self.default_axis = None

        self.fig_dimensions = [figsize[1],figsize[0]] # rows, columns

        self.fig = plt.figure(figsize=figsize)
        self.grid = plt.GridSpec(figsize[1], figsize[0], hspace=hspace, wspace=wspace)

    def add_axis(self,span=None,set_default=False,sharex=None,sharey=None):

        if span is None:
            span = self.grid[:,:]
            set_default = True

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
            print('Please specify a valid inset axis location')
            return None

        return inset_axes(a, width=height, height=height, loc=loc,borderpad=borderpad)

    def line(self,xvals,yvals,axis=None,c='k',lw=2,ls='-',label=None):
        a = self.set_axis(axis)
        a.plot(xvals,yvals,c=c,lw=lw,ls=ls,label=label)

    def line_bold(self,xvals,yvals,axis=None,c='k',lw=2,ls='-',label=None):
        a = self.set_axis(axis)
        a.plot(xvals,yvals,c='w',lw=lw*2,ls=ls,label=label)
        a.plot(xvals,yvals,c=c,lw=lw,ls=ls,label=label)

    def shade(self,xvals,y0,y1,axis=None,c='k',alpha=0.5):
        a = self.set_axis(axis)
        a.fill_between(xvals,y0,y1,color=c,alpha=alpha)

    def scatter(self,xvals,yvals,axis=None,c='k',s=10,marker='o',label=None):
        a = self.set_axis(axis)
        a.scatter(xvals,yvals,c=c,s=s,lw=0,marker=marker,label=label,rasterized=True)

    def scatter_coloured(self,xvals,yvals,cvals,vmin=None,vmax=None,alpha=1,axis=None,cmap='YlOrRd',s=10,marker='o',label=None):
        a = self.set_axis(axis)
        if vmin is None:
            vmin = np.amin(cvals)      
        if vmax is None:
            vmax = np.amax(cvals)      
        return a.scatter(xvals,yvals,c=cvals,cmap=cmap,vmin=vmin,vmax=vmax,s=s,lw=0.4,edgecolor='gray',rasterized=True,alpha=alpha)

    def scatter_hollow(self,xvals,yvals,axis=None,c='k',s=10,marker='o',label=None):
        a = self.set_axis(axis)
        a.scatter(xvals,yvals,edgecolors=c,s=s,marker=marker,facecolors='none',rasterized=True)

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
# python parameters
import matplotlib.colorbar as colorb
from matplotlib import colors
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

# my models
import models.sigma0_xsections as sig0
import models.general_parameters as gp
import models.relic_density_calc as ohm
import models.radiative_factorization as hp
import models.pdf_integration as qcd



def cb_plot3x1(figure_name, sigmaz, x_arr, y_arr, omegarelic_mass, x_ohm, y_ohm, dmlist, gr,
               cmap = 'twilight', fsize = 18, folder = '../figures'):

    Mmed_arr = x_arr
    mx_arr = y_arr
    for fignumber, dmname in enumerate(dmlist):
        print(f'doing {dmname}')
        
        Mmed_grid, mx_grid = np.meshgrid(x_arr, y_arr) # grid formation for plot
        # Set up figure and image grid  
        if fignumber == 0:
            fig = plt.figure(figsize=(20, 6))
            ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,3),
                    axes_pad=(0.30, 0.0),
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.10,
                    aspect=False
                    )
                        # Colorbar
        
        ## Plot it in a color bar
        im = ax[fignumber].pcolormesh(Mmed_grid, mx_grid, sigmaz[f'{dmname}']*gp.brn, norm=colors.LogNorm(vmin=1E1, vmax=1E15), cmap= cmap, rasterized=True) ## heat map of xsec
        ax[0].cax.cla()
        cb = colorb.Colorbar(ax[0].cax, im)
        ax[0].cax.toggle_label(True)
        cb.solids.set_rasterized(True)
        cb.set_label(r'$\sigma_{tot}$ [fb]', fontsize = fsize, loc='top')  ## color bar label 
        cb.ax.tick_params(labelsize=fsize+4)


        # ######################################################################################
        # M/2 line #
        x1, y1 = [Mmed_arr[0], Mmed_arr[-1]], [Mmed_arr[0]/2, Mmed_arr[-1]/2]
        ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')

        textstr = '\n'.join((
            r'$g_{SM} = %1.2f$, $g_{\chi} = %1.0f$' %(gr, gp.gx0),
            ))
        ax[0].plot([], [], label=textstr, color = 'None')


        ## Plot Labels ##
        if fignumber == 0:
            ab, = plt.plot(x1, y1, linewidth=3, color='gray', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
            ac, = plt.plot(0, 0, linewidth=2.5, color='red', linestyle='dashed', label=r'$2{m_\chi } = 0.8{M_{med} }$')
            ad, = plt.plot([], [], label=textstr, color = 'None')
            #ac, = plt.plot([], [],  color='lightskyblue', linestyle='--', label=r'CMS/ATLAS $Z^{\prime}$ limit')
            ax[fignumber].tick_params(axis="y", labelsize=fsize+3)

        ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)
        ax[fignumber].tick_params(axis="x", labelsize=fsize+3)

        ####################### RELIC DENSITY CONTOUR  #######################################
        # Define a grid
        Mmed_grid_ohm, mx_grid_ohm = np.meshgrid(x_ohm, y_ohm) # grid formation for plot
        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[f'{dmname}'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[f'{dmname}'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')

        @np.vectorize
        def contorno_090(Mmed, mx):
            if (2*mx <= (0.8 * Mmed)): 
                res = 10
            else:
                res = 0.00001

            return res

        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, contorno_090(Mmed_grid_ohm, mx_grid_ohm), ohm.planckdata1, colors='red', linewidths=2.5, linestyles='dashed')
        labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'

        # ######################################################################################
        
        ax[fignumber].set_xlim(0, 5.0) ## plot 'resolution'
        ax[fignumber].set_ylim(0, 2.519) ## plot 'resolution'
        
        # # ### FORMAÇÃO DA LEGENDA (PERFEITA)
        #plt.subplots_adjust(right=0.76)
        artists1, labels1 = countour_relic.legend_elements()
        labels1[0] = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)' 
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        artists1.extend(current_handles)
        labels1.extend(current_labels)
        artists1[0]._hatch = "\\\\\\\ "
        #artists1[0].xy = [x00, y00]
        artists1[0]._linestyle = 'dashed'
        artists1[0]._dashes = True
        artists1[0]._linewidth = 2.0
        artists1[0]._edgecolor = (0.0, 0.0, 0.0, 1.0)
        LEG =  ax[0].legend(artists1, labels1, loc='upper left', fontsize = fsize-2.5)

        #get the extent of the largest box containing all the axes/subplots
        if fignumber == 2:
            extents = np.array([a.get_position().extents for a in ax])  #all axes extents
            bigextents = np.empty(4)   
            bigextents[:2] = extents[:,:2].min(axis=0)
            bigextents[2:] = extents[:,2:].max(axis=0)

            #text to mimic the x and y label. The text is positioned in the middle 
            labelpad=0.02  #distance between the external axis and the text
            xlab_t = fig.text(bigextents[2], bigextents[1]-0.15, r'$Z^{\prime}$ mass [TeV]',
                horizontalalignment='right', verticalalignment = 'bottom', size = fsize+2.5)
            ylab_t = fig.text( bigextents[0]*0.7, bigextents[0]*7, r'DM mass, $m_{\chi}$ [TeV]',
                rotation='vertical', horizontalalignment = 'center', verticalalignment = 'top', size = fsize+2.5)

        if fignumber == 2: 
            fig.set_dpi(72)
            fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")

            ## Plot Labels ##
            plt.xlabel(r'$Z^{\prime}$ mass, $\sqrt{\hat s} = M_{med}$ [TeV]', fontsize = 20, loc = 'right')
            plt.ylabel(r'DM mass, $m_{\chi}$ [TeV] ', fontsize = 20, loc='top')




def cb_plot3x1(figure_name, sigmaz, x_arr, y_arr, omegarelic_mass, x_ohm, y_ohm, dmlist, gr,
               cmap = 'blues', fsize = 18, folder = '../figures'):

    Mmed_arr = x_arr
    mx_arr = y_arr
    for fignumber, dmname in enumerate(dmlist):
        print(f'doing {dmname}')
        
        Mmed_grid, mx_grid = np.meshgrid(x_arr, y_arr) # grid formation for plot
        # Set up figure and image grid  
        if fignumber == 0:
            fig = plt.figure(figsize=(20, 6))
            ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,3),
                    axes_pad=(0.30, 0.0),
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.10,
                    aspect=False
                    )
                        # Colorbar
        
        ## Plot it in a color bar
        im = ax[fignumber].pcolormesh(Mmed_grid, mx_grid, sigmaz[f'{dmname}']*gp.brn, norm=colors.LogNorm(vmin=1E1, vmax=1E15), cmap= cmap, rasterized=True) ## heat map of xsec
        ax[0].cax.cla()
        cb = colorb.Colorbar(ax[0].cax, im)
        ax[0].cax.toggle_label(True)
        cb.solids.set_rasterized(True)
        cb.set_label(r'$\sigma_{tot}$ [fb]', fontsize = fsize, loc='top')  ## color bar label 
        cb.ax.tick_params(labelsize=fsize+4)


        # ######################################################################################
        # M/2 line #
        x1, y1 = [Mmed_arr[0], Mmed_arr[-1]], [Mmed_arr[0]/2, Mmed_arr[-1]/2]
        ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')

        textstr = '\n'.join((
            r'$g_{SM} = %1.2f$, $g_{\chi} = %1.0f$' %(gr, gp.gx0),
            ))
        ax[0].plot([], [], label=textstr, color = 'None')


        ## Plot Labels ##
        if fignumber == 0:
            ab, = plt.plot(x1, y1, linewidth=3, color='gray', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
            ac, = plt.plot(0, 0, linewidth=2.5, color='red', linestyle='dashed', label=r'$2{m_\chi } = 0.8{M_{med} }$')
            ad, = plt.plot([], [], label=textstr, color = 'None')
            #ac, = plt.plot([], [],  color='lightskyblue', linestyle='--', label=r'CMS/ATLAS $Z^{\prime}$ limit')
            ax[fignumber].tick_params(axis="y", labelsize=fsize+3)

        ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)
        ax[fignumber].tick_params(axis="x", labelsize=fsize+3)

        ####################### RELIC DENSITY CONTOUR  #######################################
        # Define a grid
        Mmed_grid_ohm, mx_grid_ohm = np.meshgrid(x_ohm, y_ohm) # grid formation for plot
        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[f'{dmname}'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[f'{dmname}'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')

        @np.vectorize
        def contorno_090(Mmed, mx):
            if (2*mx <= (0.8 * Mmed)): 
                res = 10
            else:
                res = 0.00001

            return res

        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, contorno_090(Mmed_grid_ohm, mx_grid_ohm), ohm.planckdata1, colors='red', linewidths=2.5, linestyles='dashed')
        labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'

        # ######################################################################################
        
        ax[fignumber].set_xlim(0, 5.0) ## plot 'resolution'
        ax[fignumber].set_ylim(0, 2.519) ## plot 'resolution'
        
        # # ### FORMAÇÃO DA LEGENDA (PERFEITA)
        #plt.subplots_adjust(right=0.76)
        artists1, labels1 = countour_relic.legend_elements()
        labels1[0] = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)' 
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        artists1.extend(current_handles)
        labels1.extend(current_labels)
        artists1[0]._hatch = "\\\\\\\ "
        #artists1[0].xy = [x00, y00]
        artists1[0]._linestyle = 'dashed'
        artists1[0]._dashes = True
        artists1[0]._linewidth = 2.0
        artists1[0]._edgecolor = (0.0, 0.0, 0.0, 1.0)
        LEG =  ax[0].legend(artists1, labels1, loc='upper left', fontsize = fsize-2.5)

        #get the extent of the largest box containing all the axes/subplots
        if fignumber == 2:
            extents = np.array([a.get_position().extents for a in ax])  #all axes extents
            bigextents = np.empty(4)   
            bigextents[:2] = extents[:,:2].min(axis=0)
            bigextents[2:] = extents[:,2:].max(axis=0)

            #text to mimic the x and y label. The text is positioned in the middle 
            labelpad=0.02  #distance between the external axis and the text
            xlab_t = fig.text(bigextents[2], bigextents[1]-0.15, r'$Z^{\prime}$ mass [TeV]',
                horizontalalignment='right', verticalalignment = 'bottom', size = fsize+2.5)
            ylab_t = fig.text( bigextents[0]*0.7, bigextents[0]*7, r'DM mass, $m_{\chi}$ [TeV]',
                rotation='vertical', horizontalalignment = 'center', verticalalignment = 'top', size = fsize+2.5)

        if fignumber == 2: 
            fig.set_dpi(72)
            fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")

            ## Plot Labels ##
            plt.xlabel(r'$Z^{\prime}$ mass, $\sqrt{\hat s} = M_{med}$ [TeV]', fontsize = 20, loc = 'right')
            plt.ylabel(r'DM mass, $m_{\chi}$ [TeV] ', fontsize = 20, loc='top')


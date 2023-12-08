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
import datetime


def cb_plot3x1(process, figure_name, sigmaz, omegarelic_mass, dmlist, gr,
               cmap = 'twilight', fsize = 18, folder = '../figures', today = datetime.date.today().strftime('%Y-%m-%d')):


    for fignumber, dmname in enumerate(dmlist):
        print(f'doing {dmname}')
        
        Mmed_grid, mx_grid = sigmaz[dmname]['params']['Mmed'], sigmaz[dmname]['params']['mx'] # grid formation for plot
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
        im = ax[fignumber].pcolormesh(Mmed_grid, mx_grid, sigmaz[dmname]['data']*gp.brn, norm=colors.LogNorm(vmin=1E1, vmax=1E15), cmap= cmap, rasterized=True) ## heat map of xsec
        ax[0].cax.cla()
        cb = colorb.Colorbar(ax[0].cax, im)
        ax[0].cax.toggle_label(True)
        cb.solids.set_rasterized(True)
        cb.set_label(r'$\sigma_{tot}$ [fb]', fontsize = fsize, loc='top')  ## color bar label 
        cb.ax.tick_params(labelsize=fsize+4)


        # ######################################################################################
        # M/2 line #
        x1, y1 = [Mmed_grid[0][0], Mmed_grid[0][-1]], [Mmed_grid[0][0]/2, Mmed_grid[0][-1]/2]
        ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')

        textstr = '\n'.join((
            r'$g_{SM} = %1.2f$, $g_{\chi} = %1.0f$' %(gr, gp.gx0),
            ))
        ax[0].plot([], [], label=textstr, color = 'None')

        textstr = textstr + '\n' + gp.ee_qq[process]

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
        
        Mmed_grid_ohm, mx_grid_ohm = omegarelic_mass[dmname]['params']['Mmed'], omegarelic_mass[dmname]['params']['mx'] # grid formation for plot
        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')

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
        LEG.get_frame().set_alpha(0.01)

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




def cb_plot3x1_relic(process, figure_name, omegarelic_mass, dmlist, gr,
               cmap = 'Blues', fsize = 18, folder = '../figures', today = datetime.date.today().strftime('%Y-%m-%d'), ):


    for fignumber, dmname in enumerate(sig0.dmnames):
        


        Mmed_grid_ohm, mx_grid_ohm = omegarelic_mass[dmname]['params']['Mmed'], omegarelic_mass[dmname]['params']['mx'] # grid formation for plot
        
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
            fsize = 18

        ax[fignumber].set_ylim(mx_grid_ohm[0][0], mx_grid_ohm[-1][0]) ## plot 'resolution'

        im = ax[fignumber].pcolormesh(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], vmax=1, vmin=0, cmap=cmap) ##sessão de choque convertida para fb

        # Colorbar
        ax[fignumber].cax.cla()
        cb = colorb.Colorbar(ax[fignumber].cax, im, )
        ax[fignumber].cax.toggle_label(True)
        cb.set_label(r'$\Omega_{\chi} h^2 $', fontsize = fsize, loc='top')  ## color bar label 
        cb.ax.tick_params(labelsize=fsize+4)


        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')
        labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'
        #ax[fignumber].axvline(x = 2.0, color='lightskyblue', linestyle='--', label='CMS/ATLAS')
        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, omegarelic_mass[dmname]['data'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')

        @np.vectorize
        def contorno_090(Mmed, mx):
            if (2*mx <= (0.8 * Mmed)): 
                res = 1
            else:
                res = 0.001

            return res


        #countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, contorno_090(Mmed_grid_ohm, mx_grid_ohm), planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, contorno_090(Mmed_grid_ohm, mx_grid_ohm), ohm.planckdata1, colors='red', linewidths=2.5, linestyles='dashed')
        labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'


        # M/2 line #
        x1, y1 = [Mmed_grid_ohm[0][0], Mmed_grid_ohm[0][-1]], [Mmed_grid_ohm[0][0]/2, Mmed_grid_ohm[0][-1]/2]
        ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
        
        textstr = r"""$g_{SM} = %1.2f$, $g_{\chi} = %1.0f$""" %(gr, gp.gx0)

        textstr = textstr + '\n' + gp.ee_qq[process]
        ## Plot Labels ##
        if fignumber == 0:
            ab, = plt.plot(x1, y1, linewidth=3, color='gray', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
            ad, = plt.plot(0, 0, linewidth=2.5, color='red', linestyle='dashed', label=r'$2{m_\chi } = 0.8{M_{med} }$')
            ac, = plt.plot([], [], label=textstr, color = 'None')


        ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)
        ax[fignumber].tick_params(axis="x", labelsize=fsize+3)
        ax[0].tick_params(axis="y", labelsize=fsize+3)
        
        ### FORMAÇÃO DA LEGENDA (PERFEITA)
        #plt.subplots_adjust(right=0.76)
        ax[fignumber].set_xlim(Mmed_grid_ohm[0][0], Mmed_grid_ohm[0][-1]) ## plot 'resolution'
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
        LEG.get_frame().set_alpha(0.01)


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
            
            #fig.tight_layout()
            #plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
            fig.set_dpi(72)
            fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")
                
            ## Plot Labels ##
            #plt.title(r''+dmname+' DM production, $e^{+} e^{-} \to Z^{\prime} \to \chi \bar \chi$ '+auxLabel +'', fontsize = 20, loc = 'right')
            #plt.xlabel(r'$Z^{\prime}$ mass, $\sqrt{\hat s} = M_{med}$ [TeV]', fontsize = 20, loc = 'right')
            #plt.ylabel(r'DM mass, $m_{\chi}$ [TeV] ', fontsize = 20, loc='top')
            #plt.tick_params(axis="x", labelsize=20)
            #plt.tick_params(axis="y", labelsize=20)




def cb_1D(process, figure_name, object, dmname,
               cmap = 'twilight', fsize = 18, folder = '../figures', today = datetime.date.today().strftime('%Y-%m-%d'), ):


        Mmed_grid_ohm, mx_grid_ohm = object[dmname]['params']['Mmed'], object[dmname]['params']['mx'] # grid formation for plot
        
        # Set up figure and image grid  
        fignumber = 0
        fig = plt.figure(figsize=(7, 6))
        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,1),
                axes_pad=(0.30, 0.0),
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.10,
                aspect=False
                )
        fsize = 18

        ax[fignumber].set_ylim(0, 2.50) ## plot 'resolution'

        im = ax[fignumber].pcolormesh(Mmed_grid_ohm, mx_grid_ohm, object[dmname]['data'], cmap=cmap) ##sessão de choque convertida para fb

        # Colorbar
        ax[fignumber].cax.cla()
        cb = colorb.Colorbar(ax[fignumber].cax, im, )
        ax[fignumber].cax.toggle_label(True)
        cb.set_label(r'$\Omega_{\chi} h^2 $', fontsize = fsize, loc='top')  ## color bar label 
        cb.ax.tick_params(labelsize=fsize+4)


        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, object[dmname]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, object[dmname]['data'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')
        labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'

        countour_relic = ax[fignumber].contourf(Mmed_grid_ohm, mx_grid_ohm, object[dmname]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
        countour_relic_line = ax[fignumber].contour(Mmed_grid_ohm, mx_grid_ohm, object[dmname]['data'], ohm.planckdata1, colors='k', linewidths=2.5, linestyles='dashed')

        # M/2 line #
        x1, y1 = [Mmed_grid_ohm[0][0], Mmed_grid_ohm[0][-1]], [Mmed_grid_ohm[0][0]/2, Mmed_grid_ohm[0][-1]/2]
        ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
        
        textstr = r"""$g_{SM} = %1.2f$, $g_{\chi} = %1.0f$""" %(gp.gr0_q, gp.gx0)

        textstr = textstr + '\n' + gp.ee_qq[process]
        ## Plot Labels ##

        ab, = plt.plot(x1, y1, linewidth=3, color='gray', linestyle='solid', label=r'${m_\chi } = {M_{med} }/2$')
        ad, = plt.plot(0, 0, linewidth=2.5, color='red', linestyle='dashed', label=r'$2{m_\chi } = 0.8{M_{med} }$')
        ac, = plt.plot([], [], label=textstr, color = 'None')


        ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)
        ax[fignumber].tick_params(axis="x", labelsize=fsize+3)
        ax[0].tick_params(axis="y", labelsize=fsize+3)
        
        ### FORMAÇÃO DA LEGENDA (PERFEITA)
        #plt.subplots_adjust(right=0.76)
        ax[fignumber].set_xlim(0, 5.0) ## plot 'resolution'
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
        LEG.get_frame().set_alpha(0.01)


        #get the extent of the largest box containing all the axes/subplots

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


        
        #fig.tight_layout()
        #plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
        fig.set_dpi(72)
        fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")
            
        ## Plot Labels ##
        #plt.title(r''+dmname+' DM production, $e^{+} e^{-} \to Z^{\prime} \to \chi \bar \chi$ '+auxLabel +'', fontsize = 20, loc = 'right')
        #plt.xlabel(r'$Z^{\prime}$ mass, $\sqrt{\hat s} = M_{med}$ [TeV]', fontsize = 20, loc = 'right')
        #plt.ylabel(r'DM mass, $m_{\chi}$ [TeV] ', fontsize = 20, loc='top')
        #plt.tick_params(axis="x", labelsize=20)
        #plt.tick_params(axis="y", labelsize=20)




def cb_plot3x3_relic(process, figure_name, data_object, dmlist,
                      cmap = 'Blues', fsize = 18, folder = '../figures', today = datetime.date.today().strftime('%Y-%m-%d')):


    row = 0
    fignumber = 0
    while row < 9:
        for cp in gp.cps:
            for f, dmname in enumerate(dmlist):


                # Define a grid
                gr_grid_ohm, gx_grid_ohm = data_object[dmname][cp]['params']['gr'], data_object[dmname][cp]['params']['gx'] 
                mx = data_object[dmname][cp]['params']['mx']
                Mmed = data_object[dmname][cp]['params']['Mmed']


                # Set up figure and image grid  
                if fignumber == 0:
                    fig = plt.figure(figsize=(19, 17))
                    ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(3,3),
                        axes_pad=(0.6, 0.5),
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.10,
                        aspect=False
                        )
                    fsize = 18 
                

                im = ax[fignumber].pcolormesh(gr_grid_ohm, gx_grid_ohm, data_object[dmname][cp]['data'], vmin=0, vmax=1, cmap='Blues', rasterized=True) ## heat map of xsec
                countour_relic = ax[fignumber].contourf(gr_grid_ohm, gx_grid_ohm, data_object[dmname][cp]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
                countour_relic_line = ax[fignumber].contour(gr_grid_ohm, gx_grid_ohm, data_object[dmname][cp]['data'], ohm.planckdata1, colors='k', linewidths=2, linestyles='dashed', rasterized=True)
                labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'


                ax[fignumber].set_ylim(1E-3, 1) ## plot 'resolution'
                ax[fignumber].set_xlim(1E-3, 1) ## plot 'resolution'
                        

                # Colorbar
                ax[fignumber].cax.cla()
                cb = colorb.Colorbar(ax[fignumber].cax, im, )
                ax[fignumber].cax.toggle_label(True)
                cb.set_label(r'$\Omega_{\chi} h^2 $', fontsize = fsize, loc='top')  ## color bar label 
                cb.ax.tick_params(labelsize=fsize+4)


                fig.set_dpi(72)
                    

                #x1, y1 =  [gr[0], 1], [gx[0], 1]
                #ab, = ax[fignumber].plot(x1, y1, linewidth=1, color='gray', linestyle='dashed', label=r'${g_r} = {g_\chi}$')
                lb, = ax[fignumber].plot(0.2, 0.2, linewidth=0, color=None, label=r'$m_{\chi} = %1.3f$ TeV' '\n' r'$M_{med} = %1.1f$ TeV' %(mx, Mmed))

                textstr = '\n'.join((
                r'$m_{\chi}=%1.3f$ TeV' % (mx, ),
                r'$M_{med} = %1.1f$ TeV' %(Mmed, )))
                #ax[fignumber].axvline(x = 2.0, color='lightskyblue', linestyle='--', label='CMS/ATLAS')


                textstr = textstr + '\n' + gp.ee_qq[process]
                ## Plot Labels ##
                if fignumber == 0:
                    #ab, = plt.plot(x1, y1, linewidth=1, color='gray', linestyle='dashed', label=r'${g_r} = {g_\chi}$')
                    ac, = plt.plot([], [], label=textstr, color = 'None')

                # DM subplot names
                ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)


                ax[fignumber].tick_params(axis="x", labelsize=fsize)
                ax[fignumber].tick_params(axis="y", labelsize=fsize)


                ### FORMAÇÃO DA LEGENDA (PERFEITA)
                plt.subplots_adjust(right=0.76)
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
                LEG =  ax[0].legend(artists1, labels1, loc='upper right', fontsize = fsize-6.5)


                #get the extent of the largest box containing all the axes/subplots
                if fignumber == 8:
                    extents = np.array([a.get_position().extents for a in ax])  #all axes extents
                    bigextents = np.empty(4)   
                    bigextents[:2] = extents[:,:2].min(axis=0)
                    bigextents[2:] = extents[:,2:].max(axis=0)

                    #text to mimic the x and y label. The text is positioned in the middle 
                    labelpad=0.02  #distance between the external axis and the text
                    xlab_t = fig.text(bigextents[2]-0.12, bigextents[1]-0.05, r'$g_{\chi}$',
                        horizontalalignment='right', verticalalignment = 'bottom', size = fsize+3.5)
                    ylab_t = fig.text( bigextents[0]*0.6, bigextents[0]*7, r'$g_{r(SM)}$',
                        rotation='vertical', horizontalalignment = 'center', verticalalignment = 'top', size = fsize+3.5)
                    
                    

                if fignumber == 8:
                    
                    #fig.tight_layout()
                    #plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
                    fig.set_dpi(72)
                    fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")
                        

                    plt.ylabel(r'Dark Z^{\prime} coupling, $g_{\chi}$', fontsize = 20, loc='top')
                    plt.xlabel(r'$SM Z^{\prime} coupling, g_r$', fontsize = 20, loc = 'right')
                    plt.tick_params(axis="x", labelsize=20)
                    plt.tick_params(axis="y", labelsize=20)
                fignumber += 1
            row += 3    






def cb_plot3x3(process, figure_name, data_object, data_relic, dmlist,
                      cmap = 'twilight', fsize = 18, folder = '../figures', today = datetime.date.today().strftime('%Y-%m-%d')):

    row = 0
    fignumber = 0
    while row < 9:
        for cp in gp.cps:
            for f, dmname in enumerate(dmlist):
                
                # Define a grid
                gr_grid, gx_grid =  data_object[dmname][cp]['params']['gr'], data_object[dmname][cp]['params']['gx'] 
                gr_grid_ohm, gx_grid_ohm = data_relic[dmname][cp]['params']['gr'], data_relic[dmname][cp]['params']['gx'] 
                
                mx = data_object[dmname][cp]['params']['mx']
                Mmed = data_object[dmname][cp]['params']['Mmed']
         
                # Set up figure and image grid  
                if fignumber == 0:
                    fig = plt.figure(figsize=(19, 17))
                    ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(3,3),
                        axes_pad=(0.6, 0.5),
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.10,
                        aspect=False
                        )
                    fsize = 18 
                
                #sig0funcs = SFV(dmname).sig0
                #sighat = RadFact(sig0funcs).HPh_hsig
                #sigmamap = Intxgamma(sighat)

                im = ax[fignumber].pcolormesh(gr_grid, gx_grid, data_object[dmname][cp]['data']*gp.brn, norm=colors.LogNorm(vmin=1E1, vmax=1E15), cmap='twilight', rasterized=True) ## heat map of xsec
                
                ax[fignumber].set_ylim(0.001, 1) ## plot 'resolution'
                ax[fignumber].set_xlim(0.001, 1) ## plot 'resolution'
                #ax[fignumber].set_xscale('log') ## plot 'resolution'
                #ax[fignumber].set_yscale('log') ## plot 'resolution'

                # Colorbar
                ax[fignumber].cax.cla()
                cb = colorb.Colorbar(ax[fignumber].cax, im, )
                ax[fignumber].cax.toggle_label(True)
                cb.solids.set_rasterized(True)
                cb.set_label(r'$\sigma_{tot}$ [fb]', fontsize = fsize, loc='top')  ## color bar label 
                cb.ax.tick_params(labelsize=fsize+4)


                fig.set_dpi(72)



                x1, y1 =  [gr_grid[0][0], 1], [gx_grid[0][0], 1]
                #ab, = ax[fignumber].plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${g_r} = {g_\chi}$')
                lb, = ax[fignumber].plot(0.2, 0.2, linewidth=0, color=None, label=r'$m_{\chi} = %1.3f$ TeV' '\n' r'$M_{med} = %1.1f$ TeV' %(mx, Mmed))

                textstr = '\n'.join((
                r'$m_{\chi}=%1.3f$ TeV' % (mx, ),
                r'$M_{med} = %1.1f$ TeV' %(Mmed, )))
                textstr = textstr + '\n' + gp.ee_qq[process]


                ## Plot Labels ##
                if fignumber == 0:
                    #ab, = plt.plot(x1, y1, linewidth=3, color='grey', linestyle='solid', label=r'${g_r} = {g_\chi}$')
                    ac, = plt.plot([], [], label=textstr, color = 'None')
                # DM subplot names
                ax[fignumber].set_title(f'{dmname} DM', fontsize = fsize+5)
                ax[fignumber].tick_params(axis="x", labelsize=fsize)
                ax[fignumber].tick_params(axis="y", labelsize=fsize)


                ####################### RELIC DENSITY CONTOUR  #######################################
                countour_relic = ax[fignumber].contourf(gr_grid_ohm, gx_grid_ohm, data_relic[dmname][cp]['data'], ohm.planckdata2[0], colors='none', hatches=['\\\\'])
                countour_relic_line = ax[fignumber].contour(gr_grid_ohm, gx_grid_ohm, data_relic[dmname][cp]['data'], ohm.planckdata1, colors='k', linewidths=2, linestyles='dashed')
                labelOHM = r'${\Omega _\chi }{h^2} \geq 0.120$ (PLANCK 2018)'

                # ######################################################################################

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
                LEG =  ax[0].legend(artists1, labels1, loc='lower left', fontsize = fsize-4.5)


                #get the extent of the largest box containing all the axes/subplots
                if fignumber == 8:
                    extents = np.array([a.get_position().extents for a in ax])  #all axes extents
                    bigextents = np.empty(4)   
                    bigextents[:2] = extents[:,:2].min(axis=0)
                    bigextents[2:] = extents[:,2:].max(axis=0)

                    #text to mimic the x and y label. The text is positioned in the middle 
                    labelpad=0.02  #distance between the external axis and the text
                    xlab_t = fig.text(bigextents[2]-0.035, bigextents[1]-0.045, r'$g_{\chi}$',
                        horizontalalignment='right', verticalalignment = 'bottom', size = fsize+3.5)
                    ylab_t = fig.text( bigextents[0]*0.6, bigextents[0]*7, r'$g_{r(SM)}$',
                        rotation='vertical', horizontalalignment = 'center', verticalalignment = 'top', size = fsize+3.5)
                    
                

                if fignumber == 8:
                    fig.set_dpi(72)
                    fig.savefig(f'{folder}/{figure_name}.pdf', dpi=72, bbox_inches = "tight")
    
                    ## Plot Labels ##
                    plt.ylabel(r'Dark Z^{\prime} coupling, $g_{\chi}$', fontsize = 20, loc='top')
                    plt.xlabel(r'$SM Z^{\prime} coupling, g_r$', fontsize = 20, loc = 'right')
                    plt.tick_params(axis="x", labelsize=20)
                    plt.tick_params(axis="y", labelsize=20)
                fignumber += 1
            row += 3    

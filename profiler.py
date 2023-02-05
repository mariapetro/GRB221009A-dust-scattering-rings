import os
from astropy.visualization import hist
import astropy.io.fits as pyfits
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
import scipy.optimize 
from IPython.core.display import display, HTML
import astropy.wcs as wcs
import matplotlib.pyplot as plt
import matplotlib as mpl
from importlib import reload
from scipy import signal
import warnings
from astropy.io import fits
import scipy.stats as st
import emcee
import corner
from copy import deepcopy
warnings.filterwarnings("ignore")


################################
# GENERAL FUNCTION DEFINITIONS #
###############################

poly_fit='PSF'        # Options: Poly or PSF
ring_fit="Lorentzian"  # Options: "Gauss" or "Lorentzian"

def poly(x,d,f,p):
    if poly_fit=='Poly':
        return d*x**(-p)+10**f
    if poly_fit=='PSF':
        #beta=1.31 fixed parameter for Swift/XRT PSF
        r_c=3.72/60.
        W=0.075
        sigma=7.42/60.
        return d*(W*np.exp(-x**2./(2.*sigma**2))+(1-W)*(1.+(x/r_c)**2.)**(-p))+10**f

def ring_profile(x,a,b,c):
    if ring_fit=='Gauss':
        return a*np.exp(-(x-b)**2/(2*c**2))
    if ring_fit=='Lorentzian':
        return a*1./(np.pi*c)*(c**2/((x-b)**2.+c**2))

def ffunc(x,*g_par):
    add=0.  
    for k in range(1,len(g_par)-2,3):
        add+=ring_profile(x,g_par[k],g_par[k+1],g_par[k+2])
    return poly(x,g_par[0],g_par[-2],g_par[-1])+add

def dist(center_x, center_y, x, y):
    radius = (center_x - x) ** 2 + (center_y - y) ** 2
    return radius ** (1/2)*(60)  

def ring_rad(t,d_layer_norm,t0):
    z = 0.151
    d_source = 726.5 # Mpc
    return 8.3*(d_layer_norm/10**(-6.))**(-0.5)*(d_source/100.)**(-1./2.)*((t-t0)*(3600.*24.)/(10**4.))**(1./2.)*(1.+z)**(-0.5) #arcmin

def d_dist(theta,t,t_0):
    z = 0.151
    d_source = 726.5 # Mpc
    return (8.3/theta)**2*(d_source/100)**(-1)*((t-t_0)*(3600.*24.)/10**4)*(1+z)**(-1)*10**(-6)


def time_reader(file):
    
    dfits = pyfits.open(file)
    data = dfits[1].data
    
    MJD_ref = dfits[1].header['MJDREFF']+dfits[1].header['MJDREFI']
           
    time=[]
    time_err=[]
    
    quantiles=[0.16, 0.5, 0.84] 
    q=[]
    
    for k in range(0,len(quantiles)):
        q.append(np.quantile(data["TIME"]/(24*3600), quantiles[k]))
 
    time.append(q[1]+MJD_ref)
    time_err.append([q[1]-q[0],q[2]-q[1]])
    
    return time, time_err
###############################################
# CREATION OF RADIAL PROFILES FROM XRT IMAGES # 
###############################################
def ring_builder(file_names, file, file_sky, file_ex):     
        
    print('Analyzing image:', file)
    
    dfits = pyfits.open(file)
    data = dfits[1].data

    dfits_sky = pyfits.open(file_sky)
    data_sky = dfits_sky[0].data

    hdulist = fits.open(file_sky)
    w = wcs.WCS(hdulist[0].header, hdulist)
    hdulist.close()

    dfits_ex = pyfits.open(file_ex)
    data_ex = dfits_ex[0].data

    pixscale = 6.548089E-04 * 3600
    pixscale2 = 6.548089E-04 * 60 

    if file_names == 'sw006_4':
        Xburst = 288.2645833
        Yburst = 19.7737500 
    elif file_names == 'sw008_1':
        Xburst = 288.2633333
        Yburst = 19.7733 
    elif file_names == 'sw008_2':
        Xburst = 288.2650
        Yburst = 19.7725278
    else:
        Xburst = 288.2660417
        Yburst = 19.7727388 

    radec_all=np.zeros(shape=(2,1))
    xy_all=np.zeros(shape=(2,1))
    dd_all=np.zeros(shape=(2,1))

    time, time_err = time_reader(file)

    hdulist = fits.open(file_sky)
    w = wcs.WCS(hdulist[0].header, hdulist)
    hdulist.close()

    w.all_pix2world([1, 2, 3], [1, 1, 1], 1)

    x = data['X']
    y = data['Y']
    RADEC = w.all_pix2world(x, y, 1)

    mask = (data['PI'] > 100) & (data['PI'] < 1000)  # keep hard energies
    xm = x[mask]
    ym = y[mask]

    dd = dist(Xburst, Yburst, RADEC[0], RADEC[1])

    xlist = np.linspace(1.0, 1000.0, 1000)
    ylist = np.linspace(1000.0, 1.0, 1000)

    X, Y = np.meshgrid(xlist, ylist)
    RADEC_ex = w.all_pix2world(X, Y, 1)

    dd_ex = dist(Xburst, Yburst, RADEC_ex[0],RADEC_ex[1])

    zData = np.reshape(dd_ex, (1, np.size(dd_ex)))

    data_ex_1D0 = np.reshape(data_ex, (1, np.size(data_ex)))
    data_ex_1D = data_ex_1D0[0] 
    dd_ex1 = zData[0]

    y_hist, x_hist, pol = hist(dd[dd>0.18], bins=100, color='red', histtype='stepfilled', density=False,alpha=0.5)
    x_mean=np.array([(x_hist[i+1]+x_hist[i])/2. for i in range(0,len(x_hist)-1)])
    area = np.pi*(x_hist[1:]**2 - x_hist[0:-1]**2)

    exp_hist = np.zeros(len(x_hist)-1)

#   creating a histogram corrected for the exposure map    
    for k in range(len(x_hist)-1):
        mask = (dd_ex1>=x_hist[k]) & (dd_ex1<x_hist[k+1])
        exp_dum = data_ex_1D[mask]
        arrea_dum = pixscale2**2
        exp_hist[k] = np. sum(exp_dum) * arrea_dum

 
    mask2 = (x_mean<17.)
    exp_hist = exp_hist[mask2]
    y_hist = y_hist[mask2]
    x_mean = x_mean[mask2]
    y_exp = y_hist/exp_hist
    y_exp_err = np.sqrt(y_hist)/exp_hist
    inf=10**(10) 

    x_fin=[]
    y_fin=[]
    y_err=[]
    
    if file_names == 'sw004_1':    
    #   search for peaks using the find_peaks algorithm
        mask3 = (x_mean>1.)
        height_lim=np.mean(y_exp[mask3])+0.*np.std(y_exp[mask3])
        distan=10 # in bins 
        pks=find_peaks(y_exp,height=height_lim,distance=distan)

        mask4=(x_mean[pks[0]]>1.)
        pks=(pks[0][mask4]).tolist()
        heights=(y_exp[pks]).tolist()

        low_bounds=[]
        high_bounds=[]
        guess=[]

        fit_fin=[]
        pks_fin=[]
        
        for k in range(0,len(pks)):
            low_bounds.append(0.5*y_exp[pks[k]])
            low_bounds.append(0.9*x_mean[pks[k]])
            low_bounds.append(0.) 
            high_bounds.append(1.5*y_exp[pks[k]])
            high_bounds.append(1.1*x_mean[pks[k]])
            high_bounds.append(.5) 
            guess.append(y_exp[pks[k]])
            guess.append(x_mean[pks[k]])
            guess.append(0.3)

#         down_lims=[10.,*low_bounds,0.,1.3]
#         up_lims=[80.,*high_bounds,0.0011,1.8]
#         guess_cor=[46.,*guess,0.001,1.65]

        down_lims=[10.,*low_bounds,-5.,1.3]
        up_lims=[80.,*high_bounds,-3.,1.8]
        guess_cor=[46.,*guess,-4.,1.65]

        params,pconv=scipy.optimize.curve_fit(ffunc,x_mean,y_exp,bounds=(down_lims,up_lims),p0=guess_cor,maxfev=500000)

    #   apply a filter to smooth the angular histogram
        rel=signal.savgol_filter((y_exp-ffunc(x_mean,*params))/y_exp_err,5,3)

    #   find extra peaks based on the residuals    
        ext_pks=find_peaks(rel,height=3.,distance=10)
        mask5 = (ext_pks[1]['peak_heights']<inf)&(x_mean[ext_pks[0]]<17.)&(x_mean[ext_pks[0]]>1.)
        extr_pks=(ext_pks[0][mask5]).tolist()
        extr_heights=(y_exp[extr_pks]).tolist()

        for k in range(0,len(pks)):
            temp=np.where(np.array(extr_pks)==pks[k])
            if len(temp[0])!=0.:
                extr_pks.remove(pks[k])

    #     print(extr_pks)
        pks=[*pks,*extr_pks]
        heights=[y_exp[pks]]
        pks.sort()

        low_bounds=[]
        high_bounds=[]
        guess=[]

        for k in range(0,len(pks)):
            low_bounds.append(0.5*y_exp[pks[k]])
            low_bounds.append(0.8*x_mean[pks[k]])
            low_bounds.append(0.)
            high_bounds.append(1.5*y_exp[pks[k]])
            high_bounds.append(1.2*x_mean[pks[k]])
            high_bounds.append(.5)
            guess.append(y_exp[pks[k]])
            guess.append(x_mean[pks[k]])
            guess.append(0.5)

#         down_lims=[10.,*low_bounds,0.,1.]
#         up_lims=[220.,*high_bounds,0.0011,1.7]
#         guess_cor=[196.,*guess,0.001,1.65]

        down_lims=[10.,*low_bounds,-5.,1.]
        up_lims=[220.,*high_bounds,-3.,1.7]
        guess_cor=[196.,*guess,-4.,1.65]

        params_cor,pconv_cor=scipy.optimize.curve_fit(ffunc,x_mean,y_exp,bounds=(down_lims,up_lims),p0=guess_cor,maxfev=500000)

        plt.close('all')
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale("log", nonpositive='clip')
        ax.set_yscale("log", nonpositive='clip')
        plt.plot(x_mean,y_exp,'. r',label='Data')
        plt.errorbar(x_mean,y_exp,yerr=y_exp_err,alpha=0.42)
        plt.plot(x_mean,ffunc(x_mean,*params_cor),'-b',label= poly_fit+'+'+ring_fit+' '+'Fit')
        plt.plot(x_mean[pks],y_exp[pks],'m*') 
        plt.ylabel('$\\frac{Counts}{s \\cdot arcmin^2}$',fontsize=20)
        plt.xlabel('r (arcmin)',fontsize=15)
        plt.plot(x_mean, poly(x_mean,params_cor[0],params_cor[-2],params_cor[-1]),'-y')
        plt.legend()
        plt.show()

        pks_fin.append(pks)
        fit_fin.append(params_cor.tolist())
        print('# of peaks:', (len(fit_fin[-1])-3)/3)
    
    else: 
        pks_fin = []
        fit_fin = [] 
        
    x_fin.append(x_mean.tolist())
    y_fin.append(y_exp.tolist())
    y_err.append(y_exp_err.tolist())
    
    
    return x_fin, y_fin, y_err, fit_fin, pks_fin



# ##########################################################
# # DEFINITION OF COMPOSITE MODEL AND LIKELIHOOD FUNCTION #
# #########################################################
# def model1(params, t):
#     A_w = params[0]
#     slope=params[-1]
#     const=params[-2]
#     A = []
#     μ = []
#     σ = []
#     ring_profile_tot = 0.
#     for ind in range(1,len(params)-2,3):
#         A.append(params[ind])
#         μ.append(params[ind+1])
#         σ.append(params[ind+2])
#         ring_profile_tot += ring_profile(t, A[-1], μ[-1], σ[-1])
#     return poly(t,params[0],params[-2],params[-1])+ring_profile_tot

# def lnlike1(p, t, y, yerr):
#     return -0.5 * np.sum(((y - model1(p, t))/yerr) ** 2)

# def set_initial(x):
#     global _initial
#     _initial = x
    
# def lnprior1(p):
#     global _initial 
    
#     A_min=[]
#     A_max=[]
#     χ0_min=[]
#     χ0_max=[]
#     σ_min=[]
#     σ_max=[]
    
#     A_w_min=0.8*_initial[0]
#     A_w_max=1.2*_initial[0]
# #     b_min=0.5*_initial[-2]
# #     b_max=0.02

#     b_min=0.5*_initial[-2]
#     b_max=1.5*_initial[-2]
    
#     slope_min=0.8*_initial[-1]
#     slope_max=1.2*_initial[-1]

#     percentage_A_min = 1.
#     percentage_A_max = 0.8 #0.3

#     percentage_x0_min = 0.3
#     percentage_x0_max = 0.3
# #     percentage_x0_max = 0.05
    
#     percentage_σ_min = 1.
#     percentage_σ_max = 1. #0.1


#     for k in range(1,len(_initial)-2,3):
#         A_min.append(_initial[k]-percentage_A_min*_initial[k])
#         A_max.append(_initial[k]+percentage_A_max*_initial[k])
#         χ0_min.append(_initial[k+1]-percentage_x0_min*_initial[k+1])
#         χ0_max.append(_initial[k+1]+percentage_x0_max*_initial[k+1])
#         σ_min.append(_initial[k+2]-percentage_σ_min*_initial[k+2])
#         σ_max.append(_initial[k+2]+percentage_σ_max*_initial[k+2])
    
#     A_w = p[0]
#     const=p[-2]
#     slope=p[-1]
#     A = []
#     μ = []
#     σ = []    
    
#     for ind in range(1,len(p)-2,3):
#         A.append(p[ind])
#         μ.append(p[ind+1])
#         σ.append(p[ind+2])
#         if (A_w_min<A_w<A_w_max and slope_min<slope<slope_max and b_min<const<b_max and A_min[int(ind/3)] < A[-1] < A_max[int(ind/3)] and χ0_min[int(ind/3)] < μ[-1] < χ0_max[int(ind/3)] and  σ_min[int(ind/3)] < σ[-1] < σ_max[int(ind/3)]):
#             pass
#         else:
#             return -np.inf
#     return 0.0

# def lnprob1(p, x, y, yerr):
#     lp = lnprior1(p)
#     return lp + lnlike1(p, x, y, yerr) if np.isfinite(lp) else -np.inf
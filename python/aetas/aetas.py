#!/usr/bin/env python

"""AETAS.PY - Stellar Ages

"""

__authors__ = 'Joshua Povick <joshua.povick@montana.edu?'
__version__ = '20221007'  # yyyymmdd


#Astropy
import astropy
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

# Dlnpyutils and ages
from dlnpyutils.utils import bspline,mad,interp
import ages as ages

# dust_extinction
from dust_extinction.parameter_averages import CCM89,O94,F99,VCG04,GCC09,M14,F19,D22

# functools
from functools import partial

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams.update({'font.size': 25,'axes.facecolor':'w'})
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#Numpy/Scipy
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# pdb
import pdb

# tqdm 
from tqdm.notebook import tqdm

class Aetas():
    '''
    A codE to calculaTe stellAr ageS
    
    A code to calculate a star's extinction, age and mass using PARSEC isochrones with 
    Gaia (E)DR3 and 2MASS photometry.
    '''
    def __init__(self,star_data,isochrones,ext_law='CCM89',rv=3.1,teff_extrap_limit=100,debug=False):
        
        '''
        Inputs:
        ------
                    star_data: Table (pandas dataframe or astropy table)
                               Observed and calculated properties of a star(s) with the 
                               following columns:
                               
                               'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FE_H', 'FE_H_ERR',
                               'ALPHA_FE', 'ALPHA_FE_ERR', 'BP', 'G', 'RP', 'J', 'H', 'K',
                               'BP_ERR', 'G_ERR', 'RP_ERR', 'J_ERR', 'H_ERR', 'K_ERR', 
                               'DISTANCE'
                
                               'DISTANCE' must be in units of parsecs
                               
                               Nota Bene: Make sure all dtypes match the corresponding ones in 
                               isochrones
                    
                   isochrones: Table (pandas dataframe or astropy table)
                               PARSEC isochrone table with the following columns:
                               
                               'MH', 'Mass', 'delta_int_IMF', 'logAge', 'logTe', 'logg', 
                               'BPmag', 'Gmag', 'RPmag', 'Jmag', 'Hmag', 'Ksmag', 'label'
                               
                               'delta_int_IMF' is the difference in adjacent 'int_IMF' 
                               values for each isochrone (i.e. int_IMF[i+1]-int_IMF[i])
                               with the last value repeated as the difference returns 
                               one less element.
                               
                               'label' is the evolutionary phase label given by PARSEC
                               
                               Nota Bene: Make sure all dtypes match the corresponding ones in 
                               star_data
                        
                      ext_law: string, optional
                               extinction law to use. Default is CCM89.

                               Available Extinction Laws: 
                               -------------------------

                               CCM89 - Cardelli, Clayton, & Mathis 1989
                               O94 - O'Donnell 1994
                               F99 - Fitzpatrick 1999
                               F04 - Fitzpatrick 2004
                               VCG04 - Valencic, Clayton, & Gordon 2004
                               GCC09 - Grodon, Cartledge, & Clayton 2009
                               M14 - Maiz Apellaniz et al 2014
                               F19 - Fitzpatrick, Massa, Gordon, Bohlin & Clayton 2019
                               D22 - Decleir et al. 2022
                        
                           rv: float, optional
                               Rv (=Av/E(B-V)) extinction law slope. Default is 3.1 
                               (required to be 3.1 if ext_law = 'F99')
                        
            teff_extrap_limit: float
                               limit for maximum allowable temperature outside 
                               isochrone range that will be extrapolated 
                        
                        debug: bool
                               print useful information to the screen

        '''
        
        # Teff and log(g)
        self.teff = star_data['TEFF'] # temperature
        self.teff_err = star_data['TEFF_ERR'] # temperature error
        self.logg  = star_data['LOGG'] # log(g)
        self.logg_err = star_data['LOGG_ERR'] # log(g) error
        
        # Salaris corrected [Fe/H]
        sal_met = self.salaris_metallicity(star_data['FE_H'],star_data['FE_H_ERR'],
                                           star_data['ALPHA_FE'],star_data['ALPHA_FE_ERR'])
        
        self.salfeh,self.salfeh_err = sal_met[0],sal_met[1]
        
        # observed photometry
        self.obs_phot_labels = ['BP','G','RP','J','H','K']
        self.phot = 999999.0*np.ones(6)
        self.phot_err = 999999.0*np.ones(6)
        for i in range(6):
            self.phot[i] = star_data[self.obs_phot_labels[i]]
            self.phot_err[i] = star_data[self.obs_phot_labels[i]+'_ERR']
        
        # Distance modulus
        self.distmod = 5.0*np.log10(star_data['DISTANCE'])-5.0
        
        # PARSEC isochrones
        self.iso_phot_labels = ['BPmag','Gmag','RPmag','Jmag','Hmag','Ksmag']
        self.iso_interp_labels = ['BPmag','Gmag','RPmag','Jmag','Hmag','Ksmag','logg','delta_int_IMF']
        
        #isochrones = isochrones[np.argsort(isochrones['logAge'])]
        
        iso = isochrones[np.where(isochrones['MH']==self.closest(isochrones['MH'],self.salfeh))]
        iso = Table(np.copy(iso))
        iso.sort(['label','logAge'])
        self.iso = iso
        
        self.uniq_phases = np.unique(self.iso['label'])
        self.uniq_ages = np.unique(self.iso['logAge'])
        
        # dictionary of dictionaries for each label with each age as a key
        # gives min and max index of each age for each label for the sorted isochrones
        self.iso_phases_ages = {}

        for i in range(len(self.uniq_phases)):

            ages_dict = {}
            for j in range(len(self.uniq_ages)):
                ages, = np.where((self.iso['label']==self.uniq_phases[i])&
                                 (self.iso['logAge']==self.uniq_ages[j]))
                for x in ages:
                    ages_dict[self.uniq_ages[j]] = np.array([min(ages),max(ages)])

            self.iso_phases_ages[self.uniq_phases[i]] = ages_dict
                
        # Extinction
        self.rv = rv
        self.leff = np.array([0.5387,0.6419,0.7667,1.2345,1.6393,2.1757]) #BP, G, RP, J, H, K (microns)
        self.extlaw_coeff = self.extcoeff(law=ext_law,rv=self.rv)
        
        # Other
        self.debug = debug
        self.teff_extrap_limit = teff_extrap_limit 
        
    #################
    ### Utilities ###
    #################
        
    def closest(self,data,value):
        '''
        Find nearest value in array to given value.

        Inputs:
        ------
             data: array-like
                   data to search through

            value: float or int
                   value of interest

        Output:
        ------
            close: float or int
                   value in data closest to given value
        '''
        
        data = np.asarray(data)
        
        return data[(np.abs(np.subtract(data,value))).argmin()]
    
    def neighbors(self,data,value):
        '''
        Find values of two elements closest to the given value.

        Inputs:
        ------
              data: array-like
                    data to search through

             value: float or int
                    value of interest

        Output:     
        ------     
            close1: float or int
                    closest value under the given value

            close2: float or int
                    closest value over the given value
        '''
    
        data = np.asarray(data)
        close1 = data[(np.abs(np.subtract(data,value))).argmin()]
        data = data[np.where(data!=close1)]
        close2 = data[(np.abs(np.subtract(data,value))).argmin()]
    
        return close1,close2
    
    ###################
    ### Metallicity ###
    ###################
    
    def salaris_metallicity(self,metal,metal_err,alpha,alpha_err):
        '''
        Calculate the Salaris corrected metallicity (Salaris et al. 1993) using updated solar 
        parameters from Asplund et al. 2021.
        
        Inputs:
        ------
                 metal: float
                        [Fe/H] of a star
                         
             metal_err: float
                        error in [Fe/H] of a star
                     
                 alpha: float
                        [alpha/Fe] of a star
                 
             alpha_err: float
                        error in [alpha/Fe] of a star
                        
        Outputs:
        -------
                salfeh: float
                        Salaris corrected metallicity
            
            salfeh_err: float
                        error in Salaris corrected metallicity
            
        '''
        salfeh = metal+np.log10(0.659*(10**alpha)+0.341)
        salfeh_err = np.sqrt(metal_err**2+(10**alpha/(0.517372+10**alpha)*alpha_err)**2)
        
        return salfeh, salfeh_err
    
    ##################
    ### Extinction ###
    ##################
    
    def extcoeff(self,law='CCM89',rv=3.1):

        '''
        Calculate the relative extincion law coefficients for the BP, G, RP, J, H, Ks bands
        for a given Rv and extinction law.

        Input:
        -----
                    rv: float
                        Rv (=Av/E(B-V)) extinction law slope. Default is 3.1
                        
                   law: str
                        extinction law to use

                        Available Extinction Laws: 
                        -------------------------

                        CCM89 - Cardelli, Clayton, & Mathis 1989
                        O94 - O'Donnell 1994
                        F99 - Fitzpatrick 1999
                        F04 - Fitzpatrick 2004
                        VCG04 - Valencic, Clayton, & Gordon 2004
                        GCC09 - Grodon, Cartledge, & Clayton 2009
                        M14 - Maiz Apellaniz et al 2014
                        F19 - Fitzpatrick, Massa, Gordon, Bohlin & Clayton 2019
                        D22 - Decleir et al. 2022
                        
        Output:
        ------
             ext_coeff: float
                        calculated extinction coefficients for the BP, G, RP, J, H, and K bands

        '''

        leff = {'BP':0.5387,'G':0.6419,'RP':0.7667,'J':1.2345,'H':1.6393,'K':2.1757}

        # select the extinction model
        if law == 'CCM89':
            ext_model = CCM89(Rv=rv)

        elif law == 'O94':
            ext_model = O94(Rv=rv)

        elif law == 'F99':
            ext_model = F99(Rv=rv)

        elif law == 'F04':
            ext_model = F04(Rv=rv)

        elif law == 'VCG04':
            ext_model = VCG04(Rv=rv)

        elif law == 'GCC09':
            ext_model = GCC09(Rv=rv)

        elif law == 'M14':
            ext_model = M14(Rv=rv)

        elif law == 'F19':
            ext_model = F19(Rv=rv)

        elif law == 'D22':
            ext_model = D22(Rv=rv)   

        # Calculate the relative extinction coefficient
        ext_coeff_array = ext_model(np.reciprocal(self.leff*u.micron))

        return ext_coeff_array
    
    def extinction(self):
        '''
        Calulate the extinctions for the BP, G, RP, J, H, and Ks bands
        
        Input:
        -----
            #label: int
                   label for PARSEC evolutionary phase
                   
                   Available Labels:
                   ----------------
                   0 = PMS, pre main sequence
                   1 = MS, main sequence
                   2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
                   3 = RGB, red giant branch, or the quick stage of red giant for 
                   intermediate+massive stars
                   4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for 
                   intermediate+massive stars
                   5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
                   6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
                   7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for 
                   massive stars
                   8 = TPAGB, the thermally pulsing asymptotic giant branch
                   9 = post-AGB
        
        Output:
        ------
              ext: 6x2 array
                   first column is the extinction values and the second is the errors
        '''
        
        if self.debug:
            print('### Running Aetas.extinction() ###')
            print('Inputs from Aetas.__init__()')
            print('Salaris Corrected Metallicity:',self.salfeh)
            print('Temperature:',self.teff)

        # isochrone magnitude labels

        blues = np.array(['BPmag','Gmag','Gmag','Gmag','Gmag'])
        reds = np.array(['Gmag','RPmag','Jmag','Hmag','Ksmag'])
        
        # pick isochrone points with temperatures within 500 K of the star's Teff
        teffcut = np.where((self.iso['logTe']<np.log10(self.teff+500.))&
                           (self.iso['logTe']>np.log10(self.teff-500.)))
        
        iso_ = self.iso[teffcut]
        
        # check to make sure there are enough isochrone points
        if np.size(np.squeeze(teffcut))==0:
            self.ext = 999999.0*np.ones(6)
            self.ext_err = 999999.0*np.ones(6)
            return 999999.0*np.ones(6), 999999.0*np.ones(6)
    
        # get colors and errors
        obs_colors = np.delete(self.phot-self.phot[1],1)
        obs_colors[1:] = -1*obs_colors[1:]
        obs_colors_err = np.delete(np.sqrt(self.phot_err**2+self.phot_err[1]**2),1)
        
        if self.debug:
            print('Calculated Observed Colors:')
            print('Observed Colors:',obs_colors)
            print('Observed Color Errors:',obs_colors_err)
        
        # create vector of "reddening" (slightly modified from the normal definition)
        red_vec = np.delete(1-self.extlaw_coeff/self.extlaw_coeff[1],1) # Relative to G
        red_vec[0] = -1*red_vec[0]
        
        # relative extinction vector
        ext_vec = self.extlaw_coeff/self.extlaw_coeff[1] # Relative to G
        
        # calculate the intrinsic colors using a b-spline
        iso_colors = 999999.0*np.ones(5)
        iso_colors_deriv = 999999.0*np.ones(5)
        
        # determine if the Teff is in the isochrone range
        use_lgteff = np.log10(self.teff)
        if use_lgteff < np.min(iso_['logTe']) or use_lgteff > np.max(iso_['logTe']):
            use_lgteff = closest(iso_['logTe'],np.log10(self.teff))
        
        # Interpolate the color-Teff relation using a b-spline
        logTe = iso_['logTe']
        for i in range(5):
            try:
                color = (iso_[blues[i]]-iso_[reds[i]])

                bspl = bspline(logTe,color)
                iso_colors[i] = bspl(use_lgteff)
                iso_colors_deriv[i] = bspl.derivative()(use_lgteff)
                
            except:
                try:
                    bspl = bspline(logTe,color,extrapolate=True)
                    iso_colors[i] = bspl(use_lgteff)
                    iso_colors_deriv[i] = bspl.derivative()(use_lgteff)
                
                except:
                    iso_colors[i] = 999999.0
                    iso_colors_deriv[i] = 999999.0
                    
        if self.debug:
            print('Isochrone Colors:',iso_colors)
                
        # calculate the extinctions and errors
        color_diff = obs_colors-iso_colors
        color_errs = np.abs((iso_colors_deriv*self.teff_err)/(self.teff*np.log(10)))
        color_diff_err = np.sqrt(obs_colors_err**2+color_errs**2)
        
        # find bad values this should take care of bad values from the spline
        neg_cut = np.where(color_diff>0)
        
        # if all bad return bad values
        if np.size(np.squeeze(neg_cut))==0:
            if self.debug:
                print('All Colors are bad')
                print('Max Iso Teff:',10**np.nanmax(iso_['logTe']))
                print('Min Iso Teff:',10**np.nanmin(iso_['logTe']))
                print('Obs Teff:',self.teff)
            
            self.ext = 999999.0*np.ones(6)
            self.ext_err = 999999.0*np.ones(6)
            return 999999.0*np.ones(6), 999999.0*np.ones(6)
        
        # calculate the extinction value and error
        ag = np.dot(red_vec[neg_cut],color_diff[neg_cut])/np.linalg.norm(red_vec[neg_cut])**2
        ag_err = np.dot(red_vec[neg_cut],color_diff_err[neg_cut])/np.linalg.norm(red_vec[neg_cut])**2
        
        ext = 999999.0*np.ones(6)
        ext_err = 999999.0*np.ones(6)
        ext = ext_vec*ag
        ext_err = ext_vec*ag_err
        
        # chisq
        iso_colors_extincted = iso_colors+red_vec*ag
        ext_chi = sum((obs_colors-iso_colors_extincted)**2/obs_colors_err**2)
        
        self.ext = ext
        self.ext_err = ext_err
        
        if self.debug:
            print('A(G)+ Error:',ag,ag_err)
            print('All Extinctions:',ext)
            print('chisq:',ext_chi)
            print('resid:',obs_colors-iso_colors_extincted)
        
        return ext, ext_err
    
    ####################################################
    ### Mags, Log(g), Age, delta int_IMF, and Masses ###
    ####################################################
    
    def teff_2_appmags_logg(self,teff,age,label,extrap):
        '''
        Calculate the apparent magnitudes and log(g) of a star given its teff and age.
        This function also calculates the change in 'int_IMF' a star woould have, though
        this is not returned, but is stored in self.delta_int_IMF.
        
        Input:
        -----
              teff: float
                    temperature of a star
            
               age: float
                    age in Gyr of a star
            
             label: int
                    label for PARSEC evolutionary phase
                   
                    Available Labels:
                    ----------------
                    0 = PMS, pre main sequence
                    1 = MS, main sequence
                    2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
                    3 = RGB, red giant branch, or the quick stage of red giant for 
                    intermediate+massive stars
                    4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for 
                    intermediate+massive stars
                    5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
                    6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
                    7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for 
                    massive stars
                    8 = TPAGB, the thermally pulsing asymptotic giant branch
                    9 = post-AGB
                    
            
            extrap: bool
                    False: no extrapolation
                    True: extrapolate
        
        Output:
        ------
             calc_: 7x1 array
                    Calculated apparent magnitudes and log(g) of a star
        
        '''
        
        extincts = self.ext
        
        if extincts[1] > 100.:
            if self.debug:
                print('Bad extinctions replaced with 0.0')
            extincts *= 0.0
        
        if self.debug:
            print('Running Aetas.teff_2_appmags()')
            print('Teff:',teff)
            print('Extinctions:',extincts)
            print('Age: ',age)
        
        # Set up
        lgteff = np.log10(teff)
        lgage = np.log10(age*10**9)
            
#         iso = self.iso[np.where(self.iso['label']==label)]
        
        # find 2 closest ages in the ischrones
        lgage_lo,lgage_hi = self.neighbors(self.uniq_ages,lgage)
        if self.debug:
            print('[age_lo,age_hi]: ',[10**lgage_lo/10**9,10**lgage_hi/10**9])

        # younger isochrone
        aidx_lo = self.iso_phases_ages[label][lgage_lo]
        iso_lo = self.iso[int(aidx_lo[0]):int(aidx_lo[1])]
        
#         aidx_lo, = np.where(self.uniq_ages==lgage_lo)
#         iso_lo = iso[age_idx[int(aidx_lo)][0]:age_idx[int(aidx_lo)][1]]

        # older isochrone
        aidx_hi = self.iso_phases_ages[label][lgage_hi]
        iso_hi = self.iso[int(aidx_hi[0]):int(aidx_hi[1])]
    
#         aidx_hi, = np.where(self.uniq_ages==lgage_hi)
#         iso_hi = iso[age_idx[int(aidx_hi)][0]:age_idx[int(aidx_hi)][1]]

        ### Temperature Check
        extrap_lo = extrap
        extrap_hi = extrap
        self.did_extrap = 0

        if (10**min(iso_lo['logTe'])-self.teff > self.teff_extrap_limit or 
            self.teff - 10**max(iso_lo['logTe']) > self.teff_extrap_limit):

            if self.debug:
                print('outside iso_lo')
                print('Age',10**lgage_lo/10**9)
                print('max iso_lo',max(iso_lo['logTe']))
                print('min iso_lo',min(iso_lo['logTe']))
                print('Teff',np.log10(self.teff))
                print('Lower - Teff',10**min(iso_lo['logTe'])-self.teff)
                
            self.mag_logg_spl_deriv = 999999.0*np.ones(7)
            self.delta_int_IMF = 999999.0
            return 999999.0*np.ones(7)

        if (10**min(iso_hi['logTe'])-self.teff > self.teff_extrap_limit or 
            self.teff - 10**max(iso_hi['logTe']) > self.teff_extrap_limit):
            if self.debug:
                print('outside iso_hi')
                print('max iso_hi',max(iso_hi['logTe']))
                print('min iso_hi',min(iso_hi['logTe']))
                print('Teff',np.log10(self.teff))
                print('Lower - Teff',10**min(iso_hi['logTe'])-self.teff)

            self.mag_logg_spl_deriv = 999999.0*np.ones(7)
            self.delta_int_IMF = 999999.0
            return 999999.0*np.ones(7)

        ### use a b-spline to get the apparent mags, log(g), and int_IMF
        age_lo = 10**lgage_lo/10**9
        age_hi = 10**lgage_hi/10**9
        calc_lo = 999999.0*np.ones(8)
        calc_hi = 999999.0*np.ones(8)

        for i in range(len(self.iso_interp_labels)):

            ### younger age spline
#             try:
            if extrap_lo:
                self.did_extrap=1
                spl_lo = interp(iso_lo['logTe'],iso_lo[self.iso_interp_labels[i]],lgteff,
                              assume_sorted=False,extrapolate=True)

            else:
                if self.debug:
                    print('no extrap lo')
                try:
                    spl_lo = bspline(iso_lo['logTe'],iso_lo[self.iso_interp_labels[i]],nord=2)(lgteff)
                
                except:
                    pdb.set_trace()

            if i == 7 and (spl_lo <= 0.0 or np.isfinite(spl_lo)==False):
                close_teff_idx = np.where(iso_lo['logTe']==self.closest(iso_lo['logTe'],lgteff))
                spl_lo = iso_lo[self.iso_interp_labels[i]][close_teff_idx]

            if np.isfinite(spl_lo)==False:
                calc_lo[i] = 999999.0
                continue

            if i <= 5:
                calc_lo[i] = spl_lo+self.distmod+extincts[i]

            else:
                calc_lo[i] = spl_lo

            if i == 7 and calc_lo[i] < 0:
                print('Calc lo')
                print('iso teffs:',repr(iso_lo['logTe'].data))
                print('iso interps',repr(iso_lo[self.iso_interp_labels[i]].data))
                print('star teff:',lgteff)
                print('delta_int_IMF:',calc_lo[i])

#             except:
#                 calc_lo[i] = 999999.0

            ### older age spline 
#             try:
            if extrap_hi:
                self.did_extrap=1
                spl_hi = interp(iso_hi['logTe'],iso_hi[self.iso_interp_labels[i]],lgteff,
                              assume_sorted=False,extrapolate=True)

            else:
                if self.debug:
                    print('no extrap hi')
                spl_hi = bspline(iso_hi['logTe'],iso_hi[self.iso_interp_labels[i]],nord=2)(lgteff)

            if i == 7 and (spl_hi <= 0.0 or np.isfinite(spl_hi)==False):
                close_teff_idx = np.where(iso_hi['logTe']==self.closest(iso_hi['logTe'],lgteff))
                spl_hi = iso_hi[self.iso_interp_labels[i]][close_teff_idx]

            if np.isfinite(spl_hi)==False:
                calc_hi[i] = 999999.0
                continue

            if i <= 5:
                calc_hi[i] = spl_hi+self.distmod+extincts[i]

            else:
                calc_hi[i] = spl_hi

            if i == 7 and calc_hi[i] < 0:
                print('Calc hi')
                print('iso teffs:',repr(iso_hi['logTe'].data))
                print('iso interps',repr(iso_hi[self.iso_interp_labels[i]].data))
                print('star teff:',lgteff)
                print('delta_int_IMF:',calc_hi[i])

#             except:
#                 calc_hi[i] = 999999.0
    
        calc_ = 999999.0*np.ones(8)
        calc_deriv = 999999.0*np.ones(7)
        for i in range(len(self.iso_interp_labels)):
            
            # Both good interpolate
            
            if calc_lo[i] != 999999.0 and calc_hi[i] != 999999.0:
                spl_ = np.poly1d(np.squeeze(np.polyfit([age_lo,age_hi],[calc_lo[i],calc_hi[i]],1)))
            
                calc_[i] = spl_(age)
            
                if i < 7:
                    calc_deriv[i] = spl_.deriv()(age)
                    if np.isfinite(calc_deriv[i])==False:
                        pdb.set_trace()

                if i == 7 and (calc_[i]<=0. or np.isfinite(calc_[i])==False):
                    close_age_idx, = np.where([age_lo,age_hi]==self.closest([age_lo,age_hi],age))
                    calc_[i] = np.array([calc_lo[i],calc_hi[i]])[close_age_idx]
             
            # if one good use that value
            elif calc_lo[i] == 999999.0:
                calc_[i] = calc_hi[i]
                calc_deriv[i] = 0
            
            elif calc_hi[i] == 999999.0:
                calc_[i] = calc_lo[i]
                calc_deriv[i] = 0
                
            else:
                print('give up')
                
        if calc_[7] == 999999.0:
            pdb.set_trace()
                
        self.mag_logg_spl_deriv = calc_deriv # used in get_age to calculate the age error
        
        if calc_[-1]>1000.:
            pdb.set_trace()
        
        self.delta_int_IMF = calc_[-1] # store int_IMF separately
        calc_ = calc_[:-1] # delete int_IMF from the calculated values
        
        return calc_
    
    def init_guess_age(self):
        '''
        Calculate an initial edcuated guess for the age of a star by searching photomtery 
        (BP, G, RP, J, H, K) and spectroscopy (Teff, log(g)) space.
        
        Nota Bene: This does require that the extinctions are already calcuated with 
        Aetas.extinction().
             
        Output:
        ------
            age: float
                 initial guess for the age of a star 
        '''
        
        extincts = self.ext
        
        if extincts[1] > 100.:
            if self.debug:
                print('Bad extinctions replaced with 0.0')
            extincts *= 0.0
        
        if self.debug:
            print('Running Aetas.init_guess_age()')
            print('Teff:',self.teff)
            print('Extinctions:',extincts)
            
        # Array of observables
        # Gaia EDR3 and 2MASS photometry, logg, and Teff
        obs = np.append(np.append(np.copy(self.phot),self.logg),self.teff)
        obserr = np.append(np.append(np.copy(self.phot_err),self.logg_err),self.teff_err/(np.log(10)*self.teff))
        
        # Make sure the uncertainties are realistic
        obserr[0:6] = np.maximum(obserr[0:6],5e-3)  # cap phot errors at 5e-3
        obserr[6] = np.maximum(obserr[6],25) # cap teff errors at 25 K
        obserr[7] = np.maximum(obserr[7],0.05)      # cap logg errors at 0.05
        #obserr[8] = np.maximum(obserr[8],0.1)       # cap feh errosr at 0.1
            
        # Select region around the star in Teff, logg and [Fe/H]
        ind, = np.where((np.abs(10**self.iso['logTe']-self.teff) < 100) &
                        (np.abs(self.iso['logg']-self.logg) < 0.2))
        if len(ind)==0:
            ind, = np.where((np.abs(10**self.iso['logTe']-self.teff) < 300) &
                            (np.abs(self.iso['logg']-self.logg) < 0.4))
        if len(ind)==0:
            ind, = np.where((np.abs(10**self.iso['logTe']-self.teff) < 500) &
                            (np.abs(self.iso['logg']-self.logg) < 0.6))           
        if len(ind)==0:
            if self.debug:
                print('No close isochrone points.  Skipping')
            
            lgage = np.sum(np.multiply(self.iso['logAge'],self.iso['delta_int_IMF']))/np.sum(self.iso['delta_int_IMF'])            
            return 10**lgage/10**9
            
        # Get array of observables for the isochrone data
        local_iso = self.iso[ind]
        iso_colnames = np.copy(self.iso_interp_labels)[:-1]
        iso_colnames = np.append(iso_colnames,'logTe') #'BPmag','Gmag','RPmag','Jmag','Hmag','Ksmag','logg','logTe'
        
        iso_obs = np.zeros((len(ind),len(iso_colnames)),float)
        for j in range(len(iso_colnames)): iso_obs[:,j] = local_iso[iso_colnames[j]]
            
        # Extinction and distance modulus
        for j in range(6):
            # Extinction
            iso_obs[:,j] += extincts[j]
            # Distance modulus
            iso_obs[:,j] += self.distmod
            
        # Chi-squared
        chisq = np.sum((iso_obs - obs.reshape(1,-1))**2 / obserr.reshape(1,-1), axis=1)
        
        # Find best isochrone point
        bestind = np.argmin(chisq)
        resid = obs-iso_obs[bestind,:]
        
        age = 10**local_iso['logAge'][bestind]/10**9
        
        if self.debug:
            print('chisq:', chisq)
            print('resid:', resid)
        
        return age
    
    def get_age(self,label,extrap,maxfev=5000):#guess_ages=np.linspace(0.012,17.)[::10]
        '''
        Calculate Age of a star using chisq
        
        Inputs:
        -----  
                    label: int
                           PARSEC evolutionary phase label
                           
                           Available Labels:
                           ----------------
                           0 = PMS, pre main sequence
                           1 = MS, main sequence
                           2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
                           3 = RGB, red giant branch, or the quick stage of red giant for 
                           intermediate+massive stars
                           4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for 
                           intermediate+massive stars
                           5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
                           6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
                           7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for 
                           massive stars
                           8 = TPAGB, the thermally pulsing asymptotic giant branch
                           9 = post-AGB
        
                   extrap: bool
                           False: no extrapolation
                           True: extrapolate
                   
                   maxfev: int
                           maximum number of iterations before terminating the fitting
               
               guess_ages: array-like
                           ages in Gyr to use as initial guesses
        
        Outputs:
        -------
                      age: float
                           age of star in Gyr
                           
                  age_err: float
                           error for the calculated age
                           
                      chi: float
                           chisq value of the age
                           
                      rms: float
                           RMS value 
                           
            delta_int_IMF: float
                           change in int_IMF
            
        '''
        
        # initialize lists
        
        curve_ages = [] #999999.0
        curve_ages_err = [] #999999.0
        curve_chi = [] #999999.0
        curve_rms = [] #999999.0
        curve_delta_int_IMF = [] #999999.0
        
        #
        #self.guess_ages = guess_ages
        
#         self.guess_age = [self.init_guess_age()
        guess_ages = [self.init_guess_age()]
        self.guess_age = guess_ages[0]
        
        # set photometry error or 0.01 if tiny
        phot_err = np.maximum(self.phot_err,0.01)
        
        if self.debug:
            print('Running Aetas.get_age()')
            print('guess_ages:',guess_ages)


        for i in range(len(guess_ages)):
#             try:
            # calculate best fit parameters and covariance matrix
            obs_quants = np.append(np.copy(self.phot),self.logg)
            obs_quants_err = np.append(phot_err,self.logg_err)

            teff_2_appmags_logg_ = partial(self.teff_2_appmags_logg,label=label,extrap=extrap)
            popt,pcov = curve_fit(teff_2_appmags_logg_,self.teff,obs_quants,p0=guess_ages[i],
                                  method='lm',sigma=obs_quants_err,absolute_sigma=True,maxfev=maxfev)
            
            if np.isfinite(pcov)==False:
#                 print(r'PCOV problem')
                curve_ages.append(999999.0)
                curve_ages_err.append(999999.0)
                curve_chi.append(999999.0)
                curve_rms.append(999999.0)
                curve_delta_int_IMF.append(999999.0)
                
#                 pdb.set_trace()

            # populate lists
            curve_ages.append(popt[0])
            curve_mags_logg = np.asarray(teff_2_appmags_logg_(self.teff,popt[0],label=label,extrap=extrap))

            # Calculate Age error using derivatives of the mag and logg splines from self.teff_2_appmags_logg

            norm = np.linalg.norm(self.mag_logg_spl_deriv)                    
            if norm > 0.0:
                deriv_sigma = np.dot(self.mag_logg_spl_deriv/norm**2, np.append(phot_err,self.logg_err))

                curve_ages_err.append(np.sqrt(np.sum(np.square(deriv_sigma))))

            else:
                curve_ages_err.append(100.)

            if self.debug:
                print('Calc App Mags + logg:',curve_mags_logg)
                print('Observed Mags + logg:',obs_quants)
                print('Observed Mags + logg Errors:',obs_quants_err)


            curve_chi.append(sum((curve_mags_logg-obs_quants)**2/obs_quants_err**2))
            curve_rms.append(np.std(curve_mags_logg-obs_quants))
            curve_delta_int_IMF.append(self.delta_int_IMF)
            
#             if curve_delta_int_IMF > 1000.:
#                 pdb.set_trace()

#             except:
#                 # populate lists
#                 curve_age = 999999.0
#                 curve_age_err = 999999.0
#                 curve_chi = 999999.0
#                 curve_rms = 999999.0
#                 curve_delta_int_IMF = 999999.0

        #if self.debug:
        #    print('Guess Age, Age, Age Err, chisq, RMS, delta int_IMF',
        #          guess_age,curve_age,curve_age_err,curve_chi,curve_rms,curve_delta_int_IMF)
                
#         if curve_ages == 0: #np.sum(np.array([])<1e5)==0:
#             if self.debug:
#                 print('All Bad')
#             return 999999.0, 999999.0, 999999.0, 999999.0, 999999.0
        
        # find smallest chisq value and corresponding age
        idx = np.asarray(curve_chi).argmin()
        age = np.asarray(curve_ages)[idx]
        age_err = np.asarray(curve_ages_err)[idx]
        chi = np.asarray(curve_chi)[idx]
        rms = np.asarray(curve_rms)[idx]
        delta_int_IMF = np.asarray(curve_delta_int_IMF)[idx]
        
        self.age = age
        self.age_err = age_err
        self.chi = chi
        self.rms = rms
        self.delta_int_IMF = delta_int_IMF
        
#         if np.isfinite(curve_delta_int_IMF)==False:
#             pdb.set_trace()
        
        if self.debug:
            print('Guess Age,Best Fit Age, Age Error, chisq, RMS, delta_int_IMF:',
                  guess_ages,age,age_err,chi,rms,delta_int_IMF)

        return age,age_err,chi,rms,delta_int_IMF
    
    def age_2_mass(self,age,age_err,label,extrap):
        '''
        Calculate the mass of a star from its age
        
        Input:
        -----
                 age: float
                      age of a star in Gyr
                      
             age_err: float 
                      error in age of a star in Gyr
                      
               label: int
                      PARSEC evolutionary phase label
                      
                      Available Labels:
                      ----------------
                      0 = PMS, pre main sequence
                      1 = MS, main sequence
                      2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
                      3 = RGB, red giant branch, or the quick stage of red giant for 
                      intermediate+massive stars
                      4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for 
                      intermediate+massive stars
                      5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
                      6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
                      7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for 
                      massive stars
                      8 = TPAGB, the thermally pulsing asymptotic giant branch
                      9 = post-AGB
        
              extrap: bool
                      False: no extrapolation
                      True: extrapolate

        Output:
        ------
                mass: float 
                      mass of star in solar masses
            
            mass_err: float
                      error in the calculated mass of the star
        '''
        
        if self.debug:
            print('Running Aetas.age_2_mass()')
            print('Age:', age)
            print('Age Error:', age_err)
        
        if age == 999999.0:
            return 999999.0, 999999.0
        
        lgage = np.log10(age*10**9)
        
        teffcut = np.where((self.iso['logTe']<np.log10(self.teff+500.))&
                           (self.iso['logTe']>np.log10(self.teff-500.))&
                           (self.iso['label']==label))
        
        iso_ = self.iso[teffcut]
        
        if np.size(iso_) < 2:
            if self.debug:
                print('Not enough isochrone points')
            return 999999.0, 999999.0
        
        try:
            ### calculate the mass using a B-spline
            mass_spl = bspline(iso_['logAge'],iso_['Mass'])
            mass = float(mass_spl(lgage))
            mass_err = float(np.abs(mass_spl.derivative()(lgage)*age_err))
            
        except:
            try:
                # try to extrapolate
                #mass = interp(iso_['logAge'],iso_['Mass'],np.log10(age*10**9),
                #              assume_sorted=False,extrapolate=True)
                mass_spl = bspline(iso_['logAge'],iso_['Mass'],extrapolate=True)
                mass = float(mass_spl(lgage))
                mass_err = float(np.abs(mass_spl.derivative()(lgage)*age_err))

            except:
                mass = 999999.0
                mass_err = 999999.0
                
        if np.isfinite(mass)==False:
            if self.debug:
                print('Nonfinite Mass')
            return 999999.0, 999999.0
        
        if self.debug:
            print('Mass:',mass)
            print('Mass Error:', mass_err)
        
        return mass, mass_err
    
    def weighted_age_mass(self):
        '''
        Calculate the age and mass (plus chi, rms, and delta_int_IMF) weighted by the change in
        int_IMF (delta_int_IMF).
        
        Output:
        ------
                      wgt_age: float
                               age weighted by the change in the int_IMF in Gyr
                               
                  wgt_age_err: float
                               error in age weighted by the change in the int_IMF in Gyr
                               
                      wgt_chi: float
                               chi weighted by the change in the int_IMF
                      
                      wgt_rms: float
                               rms weighted by the change in the int_IMF
                      
            wgt_delta_int_IMF: float
                               change in int_IMF weighted by the change in the int_IMF
            
                     wgt_mass: float
                               mass weighted by the change in the int_IMF in Msun
                     
                 wgt_mass_err: float
                               error in mass weighted by the change in the int_IMF in Msun
                 
        '''
        
        if self.debug:
            print('Running Aetas.weighted_age_mass()')
        
        label_age = 999999.0*np.ones(len(self.uniq_phases))
        label_age_err = 999999.0*np.ones(len(self.uniq_phases))
        label_chi = 999999.0*np.ones(len(self.uniq_phases))
        label_rms = 999999.0*np.ones(len(self.uniq_phases))
        label_delta_int_IMF = 999999.0*np.ones(len(self.uniq_phases))
        label_mass = 999999.0*np.ones(len(self.uniq_phases))
        label_mass_err = 999999.0*np.ones(len(self.uniq_phases))
        
        # run Aetas.get_age() for each label
        for i in range(len(self.uniq_phases)):
            get_age_result = self.get_age(label=self.uniq_phases[i],extrap=False)
            label_age[i],label_age_err[i],label_chi[i],label_rms[i],label_delta_int_IMF[i] = get_age_result
            label_mass[i],label_mass_err[i] = self.age_2_mass(label_age[i],label_age_err[i],
                                                              label=self.uniq_phases[i],extrap=False)
            
        # check for good values
#         good, = np.where((np.isin(label_age,np.asarray(self.guess_age))==False)&(label_age!=999999.0))
        good, = np.where(label_age!=999999.0)
        
        # if no good values try extrapolating
        if np.size(good)==0:
            
            if self.debug:
                print('No good ages. Trying to extrapolate.')
            
            label_age = 999999.0*np.ones(len(self.uniq_phases))
            label_age_err = 999999.0*np.ones(len(self.uniq_phases))
            label_chi = 999999.0*np.ones(len(self.uniq_phases))
            label_rms = 999999.0*np.ones(len(self.uniq_phases))
            label_delta_int_IMF = 999999.0*np.ones(len(self.uniq_phases))
            label_mass = 999999.0*np.ones(len(self.uniq_phases))
            label_mass_err = 999999.0*np.ones(len(self.uniq_phases))
            
            for i in range(len(self.uniq_phases)):
                get_age_result = self.get_age(label=self.uniq_phases[i],extrap=True)
                label_age[i],label_age_err[i],label_chi[i],label_rms[i],label_delta_int_IMF[i] = get_age_result
                label_mass[i],label_mass_err[i] = self.age_2_mass(label_age[i],label_age_err[i],
                                                                  label=self.uniq_phases[i],extrap=True)
                
            # check for good values
#             good, = np.where((np.isin(label_age,np.asarray(self.guess_age))==False)&(label_age!=999999.0))
            good, = np.where(label_age!=999999.0)
            
            if np.size(good)==0:
                
                if self.debug:
                    print('No good ages after extrapolating')
                return 999999.0, 999999.0, 999999.0, 999999.0, 999999.0, 999999.0, 999999.0
            
        # calculate the weighted values 
        wgt_sum = np.sum(label_delta_int_IMF[good])
        
        wgt_age = np.dot(label_age[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_age_err = np.dot(label_age_err[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_chi = np.dot(label_chi[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_rms = np.dot(label_rms[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_delta_int_IMF = np.dot(label_delta_int_IMF[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_mass = np.dot(label_mass[good],label_delta_int_IMF[good])/wgt_sum
        
        wgt_mass_err = np.dot(label_mass_err[good],label_delta_int_IMF[good])/wgt_sum
        
#         print(wgt_sum, wgt_age, wgt_age_err, wgt_chi, wgt_rms, wgt_delta_int_IMF, wgt_mass, wgt_mass_err)
        
#         pdb.set_trace()
        
        return wgt_age, wgt_age_err, wgt_chi, wgt_rms, wgt_delta_int_IMF, wgt_mass, wgt_mass_err

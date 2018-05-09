import scipy
from scipy import ndimage 
a = open('train_stars.txt', 'r') 
import pyfits
from scipy import interpolate 
numtrain = 1000
al = a.readlines()[0:numtrain]

xgrid = arange(3, 270,0.009) 
t_o = 1/0.00027
t_max = 1/0.000000009
t_n = len(xgrid)
t_diff = (t_max-t_o)/t_n
factor = 1./3600/24.
xgrid_ac = arange(t_o*factor,t_max*factor,t_diff*factor)
diff_freq = diff(xgrid)
Per = 1/diff_freq/(10**-6)
diff_time = Per/len(xgrid)


def interpolate_to_grid(xdata, ydata,xgrid):
  f = interpolate.interp1d(xdata, ydata)
  new_ydata= f(xgrid)
  return xgrid, new_ydata

al2 = [] 
for each in al:
    al2.append(each.split()[0]) 
freqall,fluxall = [],[] 
counter = 0
for each in al2:
    print counter
    b = pyfits.open(each) 
    freq = b[1].data['FREQUENCY'] 
    flux = b[1].data['PSD'] 
    newx,newy = interpolate_to_grid(freq,flux,xgrid)
    freqall.append(newx)
    fluxall.append(newy)
    counter = counter+1

ifluxall = [] 
for each in fluxall: 
    ifluxall.append(fft.ifft(each)) 

import numpy as np
import pyfits
import pickle
import matplotlib.pyplot as plt
 
tc_fluxa_log = log(abs(real(ifluxall)))[:,0:780] #  for logg, numax and deltanu labels
tc_fluxa_log = log(abs(real(ifluxall)))[:,0:8500] # for teff label

tc_fluxa = abs(real(ifluxall))[:,0:780] 
tc_fluxa = abs(real(ifluxall))[:,0:8500] 

tc_flux = tc_fluxa_log
tc_flux_linear = tc_fluxa

tc_wavelx = [] 
tc_error = []
for each in tc_flux_linear:
    tc_error.append(1./each**0.5) # this gives best performance
    tc_wavelx.append(arange(0, len(each), 1))

error_take = array(tc_error) 
bad = isinf(error_take) 
labels = ['teff', 'logg', 'numax', 'deltanu']

nmeta = len(labels) 
teff, logg, numax, deltanu = loadtxt('train_stars.txt', usecols = (1,2,3,4), unpack =1) 
teff_train = teff[0:numtrain]
logg_train = logg[0:numtrain]
numax_train = numax[0:numtrain]
deltanu_train = deltanu[0:numtrain] 

tc_names = al2[0:numtrain] 
metaall = np.ones((len(tc_names), nmeta))
countit = np.arange(0,len(tc_flux),1)
newwl = np.arange(0,len(tc_flux),1) 
npix = np.shape(tc_flux[0]) [0]
 
dataall = np.zeros((npix, len(tc_names), 3))
for a,b,c,jj in zip(tc_wavelx, tc_flux, tc_error, countit):
    dataall[:,jj,0] = a
    dataall[:,jj,1] = b
    dataall[:,jj,2] = c

nstars = np.shape(dataall)[1]
for k in range(0,len(teff_train)):
    metaall[k,0] = teff_train[k]
    metaall[k,1] = logg_train[k]
    metaall[k,2] = numax_train[k]
    metaall[k,3] = deltanu_train[k]
 
file_in = open('training_realifft_unweighted.pickle', 'w') 
pickle.dump((dataall, metaall, labels, tc_names, al2),  file_in)
file_in.close()

import numpy as np
import pyfits
import pickle
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage 
import pyfits
a = open('stars_test.txt', 'r') 
from scipy import interpolate 
al = a.readlines()

xgrid = arange(3, 270,0.009) 
t_o = 1/0.00027
t_max = 1/0.000000009
t_n = len(xgrid)
t_diff = (t_max-t_o)/t_n
factor = 1./3600/24.
xgrid_ac = arange(t_o*factor,t_max*factor,t_diff*factor)

def interpolate_to_grid(xdata, ydata,xgrid):
  f = interpolate.interp1d(xdata, ydata)
  new_ydata= f(xgrid)
  return xgrid, new_ydata

al2 = [] 
for each in al:
    al2.append(each.split()[0]) 
numtest = len(al2)
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
for each in fluxall[0:numtest]: # used to use the full 10,000 but now only use the first 2000 
    ifluxall.append(fft.ifft(each)) 

tc_fluxa_log = log(abs(real(ifluxall)))[:,0:780] # for logg, numax and deltanu inference
tc_fluxa_log = log(abs(real(ifluxall)))[:,0:8500] # for teff inference

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

teff, logg, numax, deltanu = loadtxt('test_stars.txt', usecols = (1,2,3,4), unpack =1) 
teff_test = teff
logg_test  = logg
deltanu_test, numax_test = deltanu, numax

tc_names = al2
tc_names_test = al2
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
for k in range(0,len(tc_names)):
    metaall[k,0] = teff_test[k]
    metaall[k,1] = logg_test[k]
    metaall[k,2] = numax_test[k]
    metaall[k,3] = deltanu_test[k]
 
file_in = open('test_realifft_unweighted.pickle', 'wb') 
pickle.dump((dataall[:,0:2000,:], metaall[0:2000,:], labels, tc_names_test[0:2000], tc_names_test[0:2000]),  file_in)
file_in.close()

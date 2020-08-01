import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy
import os
import bilby
import pandas as pd

outdir = '/fred/oz008/xzhu/projects/SuperBayes/PG1302/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
label = 'red_sin_CAL'
file_names = ['PG1302_LINEAR_binned1.dat','PG1302_CRTS.dat','PG1302_ASAS-SN_DR9.dat']
flags = ['LINEAR','CRTS','ASAS-SN']
data = dict()
data['Mag'] = np.array([])
data['Magerr'] = np.array([])
data['MJD'] = np.array([])
data['Flag'] = np.array([])
for ii, file_name in enumerate(file_names):
    raw_data = np.loadtxt(file_name,skiprows=1,delimiter=',')
    data['Mag']=np.append( data['Mag'], raw_data[:,1] )
    data['Magerr']=np.append( data['Magerr'], raw_data[:,2] )
    data['MJD']=np.append( data['MJD'], raw_data[:,0] )
    data['Flag']=np.append( data['Flag'], np.repeat(flags[ii],len(raw_data[:,1])) )

data = pd.DataFrame.from_dict(data)
data.sort_values(by=['MJD']);

# parameters for the signal model: A1, A2, T0
# parameters for the red noise: CC, tau0
# white noise: nu

#main_flag = pd.unique(data['Flag'])[0]

priors = dict()
priors['A'] = bilby.core.prior.Uniform(0, 0.5, name='A', latex_label='$A$ [mag]')
priors['PHI'] = bilby.core.prior.Uniform(0, 2*np.pi, periodic_boundary=True, name='PHI', latex_label='$\\phi$ [radian]')
priors['T0'] = bilby.core.prior.Uniform(0, 10, name='T0', latex_label='$T_{0}$ [yr]')
priors['logCC'] = bilby.core.prior.Uniform(-6, 0, name='logCC', latex_label='$\\ln \\hat{\\sigma}^2$ [mag$^2$ yr$^{-1}$]')
priors['logTAU0'] = bilby.core.prior.Uniform(-4, 4, name='logTAU0', latex_label='$\\ln \\tau_{0}$ [yr]')
priors['gamma'] = bilby.core.prior.Uniform(0, 2, name='gamma', latex_label='$\\gamma$')

for flag in flags:
    priors['NU-'+flag] = bilby.core.prior.Uniform(0.1, 2.0, name='NU-'+flag, latex_label='$\\nu_{\\textrm{'+flag+'}}$')
    #if flag!=main_flag: priors['OFFSET_'+flag] = bilby.core.prior.Uniform(-10, 10, 'OFFSET_'+flag)
    priors['OFFSET-'+flag] = bilby.core.prior.Uniform(14.5, 15.5, name='OFFSET-'+flag, latex_label='$m_{\\textrm{'+flag+'}}$ [mag]')

parameters = dict.fromkeys(priors.keys())


class MultidimGaussianLikelihood(bilby.Likelihood):
    """
        A multivariate Gaussian likelihood

        """

    def __init__(self, data, parameters):
        self.data = data
        self.N = len(data['MJD'])
        self.parameters = parameters
        
        self._nu_array = np.empty(self.N)
        self._offset_array = np.empty(self.N)
        
        # Set offset always to be zero for "main" flag
        #self.main_flag = pd.unique(data['Flag'])[0]
        #self.main_flag_idx = data.index[data['Flag']==self.main_flag]
        #self._offset_array[self.main_flag_idx] = 0

        self.idx = dict()
        for flag in self.data['Flag']:
            self.idx[flag] = data.index[data['Flag']==flag]
        
        self.tauij = np.abs( np.repeat(np.asarray(self.data['MJD'])[np.newaxis],self.N,axis=0) - \
          np.repeat(np.asarray(self.data['MJD'])[np.newaxis].T,self.N,axis=1) )
        
    def map_params_to_array(self):
        
        for flag in self.data['Flag']:
            
            self._nu_array[self.idx[flag]] = self.parameters['NU-'+flag]
            #if flag!=self.main_flag: self._offset_array[self.idx[flag]] = self.parameters['OFFSET_'+flag]
            self._offset_array[self.idx[flag]] = self.parameters['OFFSET-'+flag]

    def log_likelihood(self):
        
        self.map_params_to_array()
        
        DS = np.asarray(data['Mag'])[np.newaxis] - np.asarray(self._signal_model())[np.newaxis] - np.asarray(self._offset_array)[np.newaxis]
        covm = self._get_cov_matrix()
#        normal_exponent = np.dot(DS,np.linalg.pinv(covm))
        normal_exponent = np.dot(DS,scipy.linalg.pinvh(covm))
        normal_exponent = np.dot(normal_exponent,DS.T)
        
        return np.squeeze( -0.5*normal_exponent - 0.5*(  np.log(2*np.pi)*self.N + np.linalg.slogdet(covm)[0]*np.linalg.slogdet(covm)[1]  ) , axis=1)[0]
    
    def _get_cov_matrix(self):
        return self._white_noise()+self._red_noise()
        
        
    def _signal_model(self):
        result = self.parameters['A']*np.sin(2*np.pi*self.data['MJD']/(self.parameters['T0']*365.25)+self.parameters['PHI'])
        return result
    
    def _white_noise(self):
        return np.diag(self.data['Magerr']**2*self._nu_array**2)
#        return np.diag(self.data['Magerr']**2)
    
    def _red_noise(self):
#        return 0.5*np.exp(self.parameters['logCC'])**2*np.exp(self.parameters['logTAU0'])*np.exp(-self.tauij/365.25/np.exp(self.parameters['logTAU0']))
        return 0.5*np.exp(self.parameters['logCC'])**2*np.exp(self.parameters['logTAU0'])*np.exp(-(self.tauij/(365.25*np.exp(self.parameters['logTAU0'])))**self.parameters['gamma'])

lnl = MultidimGaussianLikelihood(data,parameters)

result = bilby.run_sampler(
    likelihood=lnl,
    priors=priors,
    sampler="dynesty",
    resume=False,
    npoints=1000,
    dlogz=0.1,
    walks=30,
    outdir=outdir,
    label=label,
    plot=True,
)

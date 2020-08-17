import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy
import os
import bilby
import pandas as pd

outdir = './red_qpo'
if not os.path.exists(outdir):
    os.makedirs(outdir)
label = 'red_qpo_CA'
file_names = ['PG1302_CRTS_v0.dat','PG1302_ASAS-SN_v0.dat']
flags = ['CRTS','ASAS-SN']
data = dict()
data['Mag'] = np.array([])
data['Magerr'] = np.array([])
data['MJD'] = np.array([])
for ii, file_name in enumerate(file_names):
    raw_data = np.loadtxt(file_name,skiprows=1,delimiter=',')
    data['Mag']=np.append( data['Mag'], raw_data[:,1] )
    data['Magerr']=np.append( data['Magerr'], raw_data[:,2] )
    data['MJD']=np.append( data['MJD'], raw_data[:,0] )

data = pd.DataFrame.from_dict(data)
data.sort_values(by=['MJD']);

priors = dict()
priors['logCC'] = bilby.core.prior.Uniform(-6, 0, name='logCC', latex_label='$\\ln \\hat{\\sigma}^2$ [mag$^2$ yr$^{-1}$]')
priors['logTAU0'] = bilby.core.prior.Uniform(-4, 4, name='logTAU0', latex_label='$\\ln \\tau_{0}$ [yr]')
priors['gamma'] = bilby.core.prior.Uniform(0, 1.8, name='gamma', latex_label='$\\gamma$')
priors['Tq'] = bilby.core.prior.Uniform(0, 10, name='Tq', latex_label='$T_{q}$ [yr]')

parameters = dict.fromkeys(priors.keys())

class MultidimGaussianLikelihood(bilby.Likelihood):
    """
        A multivariate Gaussian likelihood

        """

    def __init__(self, data, parameters):
        self.data = data
        self.N = len(data['MJD'])
        self.parameters = parameters
        self.tauij = np.abs( np.repeat(np.asarray(self.data['MJD'])[np.newaxis],self.N,axis=0) - \
          np.repeat(np.asarray(self.data['MJD'])[np.newaxis].T,self.N,axis=1) )

    def log_likelihood(self):
        DS = np.asarray(data['Mag'])[np.newaxis]
        covm = self._get_cov_matrix()
        normal_exponent = np.dot(DS,scipy.linalg.pinvh(covm))
        normal_exponent = np.dot(normal_exponent,DS.T)
        return np.squeeze( -0.5*normal_exponent - 0.5*(  np.log(2*np.pi)*self.N + np.linalg.slogdet(covm)[0]*np.linalg.slogdet(covm)[1]  ) , axis=1)[0]
    
    def _get_cov_matrix(self):
        return self._white_noise()+self._red_noise()
    
    def _white_noise(self):
        return np.diag(self.data['Magerr']**2)
    
    def _red_noise(self):
        return 0.5*np.exp(self.parameters['logCC'])**2*np.exp(self.parameters['logTAU0'])*np.exp(-(self.tauij/(365.25*np.exp(self.parameters['logTAU0'])))**self.parameters['gamma'])*np.cos(2*np.pi*self.tauij/(self.parameters['Tq']*365.25))

lnl = MultidimGaussianLikelihood(data,parameters)

result = bilby.run_sampler(
    likelihood=lnl,
    priors=priors,
    sampler="dynesty",
    resume=False,
    npoints=1000,
    dlogz=0.2,
    walks=20,
    outdir=outdir,
    label=label,
    plot=True,
)

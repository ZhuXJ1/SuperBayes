# SuperBayes
Bayesian periodicity search in quasar/AGN light curves as a way to identify candidates of Supermassive binary black holes

This repository contains free codes, which come with absolutely no warranty. However, if you would like to use them and need help, please feel free to get in touch.

Codes and data that accompany the paper "Toward the unambiguous identification of supermassive binary black holes through Bayesian inference
" (https://arxiv.org/abs/2004.10944), by Xing-Jiang Zhu & Eric Thrane.

If you use this code/method in your publication, please cite Zhu & Thrane (2020). If you use any of the data sets for PG1302-102, or any other software packages used in codes presented here, please cite the original papers and if necessary add relevant acknowledgements for use of those data/software (see Zhu & Thrane 2020 for details).

Updates: The code might not immediately work with the latest version of Bilby.

AttributeError: 'MultidimGaussianLikelihood' object has no attribute ‘_marginalized_parameters'

A hack to solve the above error is to add the second line below.

    def __init__(self, data, parameters):
        self._marginalized_parameters = []
        

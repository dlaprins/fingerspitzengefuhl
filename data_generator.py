import numpy as np
import pandas as pd
from scipy.stats import random_correlation
from scipy.stats import lognorm

import matplotlib.pyplot as plt


# A class to simulate fake data to be used for modelling purposes. 


class DataGenerator: 
    """ 
    Class to generate multivariate samples from a distribution, either as a stochastic process or as a regression set:
    Currently, only the normal and lognormal distributions are implemented. 
    """

    def __init__(self, N_dim, distribution_type='normal', parameters_dict={}, correlation=None, stochastic_process=False, np_to_df=False):
        self.N_dim = N_dim
        self.distribution = distribution_type
        self.params = parameters_dict
        self.corr = correlation
        self.stochastic = stochastic_process
        self.np_to_df = np_to_df



    def consistency_check(self):
        """
        Check to see if all parameters & the correlation matrix have the correct dimension. 
        """
        N_dim = self.N_dim
        for param_name in self.params: 
            if len(self.params[param_name]) != N_dim: 
                raise ValueError('Dimensionality inconsistency for parameter {}.'.format(param_name))
        if self.corr.shape[0] != N_dim:
            raise ValueError('Dimensionality inconsistency for the correlation matrix.')



    def generate_parameters(self, intervals_dict={'mu':[], 'sigma':[]}):
        """
        Simulate parameters of the distributions from which to generate data.
        The parameters are set to the attribute self.params.

        Arguments
        --------
        intervals_dict: dictionary
            dictionary keys are the parameters for which to simulate. 
            dictionary values are lists of intervals from which to simulate. 
            For each list of intervals: if the length is the correct dimension N_dim, the specified intervals are used;
            Else, if no list of intervals is specified (i.e., an empty list), N_dim parameters are generated from [0,1];
            Else, an error is thrown.                             
        """
        for param_name, list_of_intervals in intervals_dict.items():
            N_dim = len(list_of_intervals)
            if N_dim not in [0, self.N_dim]:
                raise ValueError("The list of intervals for paramater {} is of incorrect dimension. \
                    Please use either an empty list or a list of dim {}".format(param_name, self.N_dim))
            else:
                if N_dim == 0: 
                    list_of_intervals = [ [0,1] for i in range(self.N_dim) ]
                param_i_sim = [np.random.uniform(list_of_intervals[i][0], list_of_intervals[i][1]) for i in range(self.N_dim)]
            self.params[param_name] = np.array(param_i_sim) 



    def generate_correlation(self, corr_eigenvalues = []):
        """
        Simulate correlation matrix of the multi-variate distributions from which to generate data.
        For 1-dimensional data, the correlation matrix is set to 1. 
        The correlation matrix is set to the attribute self.corr.

        Arguments
        --------
        corr_eigenvalues: list
            Eigenvalues of the correlation matrix.
            If the length is the correct dimension N_dim, the specified list;
            Else, if no list of intervals is specified (i.e., an empty list), N_dim eigenvalues are generated from [0,1];
            Else, an error is thrown.                             
        """
        if self.N_dim == 1: 
            corr_sim = np.array([[1]])
        else:    
            N_dim = len(corr_eigenvalues)
            if N_dim not in [0, self.N_dim]:
                raise ValueError("The list of correlation eigenvalues is of incorrect dimension. \
                    Please use either an empty list or a list of dim {}".format(self.N_dim))
            else: 
                if len(corr_eigenvalues) == 0:  
                    corr_eigenvalues = np.random.rand(self.N_dim)
                corr_eigenvalues = corr_eigenvalues * self.N_dim / np.sum(corr_eigenvalues)   
                corr_sim = random_correlation.rvs(corr_eigenvalues)
        self.corr = corr_sim



    def generate_data(self, N_samples):
        """
        Generate random data. 
        - The distribution of the data is given by self.distribution. 
        - The parameters of the distribution is given by self.params. 
        - The dimension of the data is given by self.N_dim. 
        - The correlation of the distribution is given by self.corr. 
        - The number of samples is given by N_samples. 
        - Whether or not the data simulates a random walk is given by self.stochastic. 
        The data is set to the attribute self.data.

        Arguments
        --------
        N_samples: int

        Return
        --------
        data: 
            depending on the boolean self.np_to_df, 
            returns either an np.array or a pd.Dataframe
            of shape (N_samples, self.N_dim)
        """

        # self.consistency_check()

        std = self.params['sigma']
        mu = self.params['mu']

        cov_sim = np.zeros_like(self.corr, dtype='float')
        for i in range(cov_sim.shape[0]):
            for j in range(cov_sim.shape[1]):
                cov_sim[i,j] = self.corr[i,j] * std[i] * std[j]

        if self.distribution == 'normal':
            data = np.random.multivariate_normal(mu, cov_sim, size = N_samples)
        elif self.distribution == 'lognormal':
            data = np.random.multivariate_normal(mu, cov_sim, size = N_samples)
            data = np.exp(data)
        else: 
            raise ValueError("Distribution type unknown. Currently implemented: 'normal', 'lognormal'.")
        
        if self.stochastic == True: 
            data = np.cumsum(data, axis=0)

        
        if self.np_to_df == True: 
            col_names = ['time_series_' + str(i) for i in range(self.N_dim)]
            data = pd.DataFrame(data, columns = col_names)
        
        self.data = data

        return data


if __name__ == '__main__':

    mu_intervals = [[0,1], [1,2], [2,3], [3,4]]
    sigma_intervals = mu_intervals[::-1]
    intervals_dict = {'mu': mu_intervals, 'sigma': sigma_intervals}
    dg = DataGenerator(N_dim = 4)
    dg.generate_correlation()
    dg.generate_parameters(intervals_dict)
    data = dg.generate_data(N_samples=10**5)
    print("Comparing generated parameters with sample parameters:")
    print(dg.params)
    print(data.mean(axis=0))
    print(data.std(axis=0))
    print("Comparing generated correlation with sample correlation:")
    print(dg.corr)
    print(np.corrcoef(data, rowvar = False))



    N_dim = 5
    intervals_dict2 = {'sigma':[]}
    dg2 = DataGenerator(N_dim = N_dim, distribution_type='normal', stochastic_process=True)
    dg2.params['mu'] = [0 for i in range(N_dim)]
    dg2.generate_parameters(intervals_dict2)
    dg2.corr = np.identity(N_dim)
    data = dg2.generate_data(N_samples=100)
    print("Plotting simulated Brownian motion:")
    for i in range(data.shape[1]):
        plt.plot(data[:,i])
    plt.show()
    

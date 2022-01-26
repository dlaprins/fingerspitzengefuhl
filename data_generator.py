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

    def __init__(self, n_dim, distribution_type='normal', parameters_dict=None, correlation=None, stochastic_process=False, np_to_df=False):
        self.counter = 0
        self.datasets = {} 
        self.n_dim = n_dim
        self.distribution = distribution_type
        self.params = parameters_dict or {}
        self.corr = correlation
        self.stochastic = stochastic_process
        self.np_to_df = np_to_df




    def consistency_check(self):
        """
        Check to see if all parameters & the correlation matrix have the correct dimension. 
        """
        n_dim = self.n_dim
        for param_name in self.params: 
            if len(self.params[param_name]) != n_dim: 
                raise ValueError('Dimensionality inconsistency for parameter {}.'.format(param_name))
        if self.corr.shape[0] != n_dim:
            raise ValueError('Dimensionality inconsistency for the correlation matrix.')


    def get_new_datakey(self):
        data_key = f'data_{self.counter}'
        return data_key

    
    def add_new_data(self, data):
        """
        Construct a dictionary consisting of the input data and all info required to generate it.
        Adds this dictionary to the datasets attribute.
        """
        full_data = {}
        full_data['data'] = data
        full_data['distribution_type'] = self.distribution
        full_data['parameters'] = self.params
        full_data['correlation'] = self.corr
        full_data['stochastic'] = self.stochastic

        datakey = self.get_new_datakey()
        self.datasets[datakey] = full_data
        self.counter += 1


    def generate_parameters(self, intervals_dict={'mu':[], 'sigma':[]}):
        """
        Simulate parameters of the distributions from which to generate data.
        The parameters are set to the attribute self.params.

        Arguments
        --------
        intervals_dict: dictionary
            dictionary keys are the parameters for which to simulate. 
            dictionary values are lists of intervals from which to simulate. 
            For each list of intervals: if the length is the correct dimension n_dim, the specified intervals are used;
            Else, if no list of intervals is specified (i.e., an empty list), n_dim parameters are generated from [0,1];
            Else, an error is thrown.                             
        """
        for param_name, list_of_intervals in intervals_dict.items():
            n_dim = len(list_of_intervals)
            if n_dim not in [0, self.n_dim]:
                raise ValueError(f"The list of intervals for parameter {param_name} is of incorrect dimension. \
                    Please use either an empty list or a list of dim {self.n_dim}")
            if n_dim == 0: 
                list_of_intervals = [ [0,1] for i in range(self.n_dim) ]
            param_i_sim = [np.random.uniform(list_of_intervals[i][0], list_of_intervals[i][1]) for i in range(self.n_dim)]
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
            If the length is the correct dimension n_dim, the specified list;
            Else, if no list of intervals is specified (i.e., an empty list), n_dim eigenvalues are generated from [0,1];
            Else, an error is thrown.                             
        """
        if self.n_dim == 1: 
            corr_sim = np.array([[1]])
        else:    
            n_dim = len(corr_eigenvalues)
            if n_dim not in [0, self.n_dim]:
                raise ValueError(f"The list of correlation eigenvalues is of incorrect dimension. \
                    Please use either an empty list or a list of dim {self.n_dim}")
            if len(corr_eigenvalues) == 0:  
                corr_eigenvalues = np.random.rand(self.n_dim)
            corr_eigenvalues = corr_eigenvalues * self.n_dim / np.sum(corr_eigenvalues)   
            corr_sim = random_correlation.rvs(corr_eigenvalues)
        self.corr = corr_sim



    def generate_data(self, n_samples):
        """
        Generate random data. 
        - The distribution of the data is given by self.distribution. 
        - The parameters of the distribution is given by self.params. 
        - The dimension of the data is given by self.n_dim. 
        - The correlation of the distribution is given by self.corr. 
        - The number of samples is given by n_samples. 
        - Whether or not the data simulates a random walk is given by self.stochastic. 
        The data and all input is added to the dictionary self.datasets.

        Arguments
        --------
        n_samples: int

        Return
        --------
        data: 
            depending on the boolean self.np_to_df, 
            returns either an np.array or a pd.Dataframe
            of shape (n_samples, self.n_dim)
        """

        self.consistency_check()

        std = self.params['sigma']
        mu = self.params['mu']

        cov_sim = np.zeros_like(self.corr, dtype='float')
        for i in range(cov_sim.shape[0]):
            for j in range(cov_sim.shape[1]):
                cov_sim[i,j] = self.corr[i,j] * std[i] * std[j]

        if self.distribution == 'normal':
            data = np.random.multivariate_normal(mu, cov_sim, size = n_samples)
            if self.stochastic == True: 
                data = np.cumsum(data, axis=0)
        elif self.distribution == 'lognormal':
            data = np.random.multivariate_normal(mu, cov_sim, size = n_samples)
            if self.stochastic == True: 
                data = np.cumsum(data, axis=0)
            data = np.exp(data)
        else: 
            raise ValueError("Distribution type unknown. Currently implemented: 'normal', 'lognormal'.")
        

        
        if self.np_to_df == True: 
            col_names = ['time_series_' + str(i) for i in range(self.n_dim)]
            data = pd.DataFrame(data, columns = col_names)
        
        self.add_new_data(data)

        return data


if __name__ == '__main__':

    mu_intervals = [[0,1], [1,2], [2,3], [3,4]]
    sigma_intervals = mu_intervals[::-1]
    intervals_dict = {'mu': mu_intervals, 'sigma': sigma_intervals}
    dg = DataGenerator(n_dim = 4)
    dg.generate_correlation()
    dg.generate_parameters(intervals_dict)
    data = dg.generate_data(n_samples = 10**5)
    print("Comparing generated parameters with sample parameters:")
    print(dg.params)
    print(data.mean(axis = 0))
    print(data.std(axis = 0))
    print("Comparing generated correlation with sample correlation:")
    print(dg.corr)
    print(np.corrcoef(data, rowvar = False))



    n_dim = 5
    intervals_dict2 = {'sigma':[]}
    dg2 = DataGenerator(n_dim = n_dim, distribution_type = 'normal', stochastic_process = True)
    dg2.params['mu'] = [0 for i in range(n_dim)]
    dg2.generate_parameters(intervals_dict2)
    dg2.corr = np.identity(n_dim)
    data = dg2.generate_data(n_samples = 100)
    print("Plotting simulated Brownian motion:")
    for i in range(data.shape[1]):
        plt.plot(data[:,i])
    plt.show()
    

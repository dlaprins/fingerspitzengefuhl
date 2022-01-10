import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers, blas

from data_generator import DataGenerator

# An exercise in using convex optimization (specifically, quadratic programming)
# to obtain the efficient frontier for an arbitrary number of (normally distributed) stocks with arbitrary correlation.



class EfficientFrontier:
    """
    Class to determine the efficient frontier, given an array of time serries of returns.
    """

    def __init__(self, data, r_f=0):
        self.data = data
        self.r_f = r_f
        self.mu_obs = np.mean(data, axis = 0)
        self.cov_obs = np.cov(data, rowvar = False)
        self.portfolios = {}
        self.fitted = False
        return



    def fit(self):
        """
        Adds the following three keys to portfolios: 
        - 'min_variance'
        - 'max_sharpe'
        - 'max_returns'
        In each case, the corresponding value consists of a dictionary with keys:
        - 'mu' : float,  (expected returns)
        - 'std' : float, (risk, expressed as standard deviation)
        - 'weights': np.array of size (n_stocks, ),  (portfolio weights, normalized to 1)

        Adds the attribute 'frontier', corresponding to a dictionary with the same keys:
        - 'mu' : list of floats
        - 'std' : list of floats
        - 'weights': list of np.arrays of size (n_stocks, )
        """

        n = self.data.shape[1]
        covariance = np.cov(self.data, rowvar = False)

        # Prepare matrices for solvers.qp, which minimizes
        # (1/2) w^T * S * w + q^T * w
        # subject to the constrains:
        # G * w <= h & A*w = b
        # The former constraint is used to enforce positve weights
        # The latter constraint is used to set the norm of the weights to 1, 
        # or to enforce expected portfolio returns, or both. 
        mu = matrix(self.mu_obs)
        S = matrix(covariance)
        G = - matrix(np.eye(n)) 
        h = matrix(0.0, (n, 1))
        q = matrix(10**(-7), (n,1))     # regulator for the numerical solution
        
        # Compute minimal variance portfolio
        a_0 = np.ones(n).reshape(-1,1).transpose()
        A_min_var = matrix(1.0, (1, n)) 
        b_min_var = matrix(1.0)
        weights_min_var = solvers.qp(S, q, G, h, A_min_var, b_min_var)['x']
        mu_min_var = blas.dot(mu, weights_min_var)
        std_min_var = np.sqrt(blas.dot(weights_min_var, S * weights_min_var))
        self.portfolios['min_variance'] = {'mu': mu_min_var, 'std': std_min_var,
                                             'weights': np.array(weights_min_var)}

        # Compute maximal Sharpe ratio portfolio
        a_1 = self.mu_obs.reshape(-1,1).transpose()
        A_max_sharpe = matrix(a_1 - r_f)
        b_max_sharpe = matrix(1.0)
        weights_max_sharpe = solvers.qp(S, q, G, h, A_max_sharpe, b_max_sharpe)['x']
        weights_max_sharpe = weights_max_sharpe / np.sum(weights_max_sharpe)
        mu_sharpe = blas.dot(mu, weights_max_sharpe)
        std_sharpe = np.sqrt(blas.dot(weights_max_sharpe, S * weights_max_sharpe))
        self.portfolios['max_sharpe'] = {'mu': mu_sharpe, 'std': std_sharpe, 
                                        'weights': np.array(weights_max_sharpe)}

        # Compute maximal expected returns portfolio
        mu_max_returns = np.max(self.mu_obs)
        idx_max_returns = np.argmax(self.mu_obs)
        std_max_returns = np.sqrt(covariance[idx_max_returns, idx_max_returns])
        weights_max_returns = np.zeros(n)
        weights_max_returns[idx_max_returns] = 1 
        self.portfolios['max_returns'] = {'mu': mu_max_returns, 'std': std_max_returns, 
                                            'weights': np.array(weights_max_returns)}

        # Compute efficient frontier: for expected returns mu in the interval [mu_min_var, mu_max_returns], 
        # use QP to find portfolio weights with given mu and minimal standard deviation 
        N = 10**3
        A_frontier = matrix(np.row_stack((a_0, a_1)))
        all_mus = [(1 - t/N) * mu_min_var   + t/N * mu_max_returns  for t in range(N+1)]
        b_mu = [np.array([[1.0], [mu]]) for mu in all_mus]
        b_mu = [matrix(b) for b in b_mu]
        weights_frontier = [solvers.qp(S, q, G, h, A_frontier, b)['x'] for b in b_mu]
        mu_frontier = [blas.dot(mu, weights) for weights in weights_frontier]
        std_frontier = [np.sqrt(blas.dot(weights, S * weights)) for weights in weights_frontier]
        self.frontier = {'mu': mu_frontier, 'std': std_frontier, 'weights': weights_frontier}

        self.fitted = True


    def plot_efficient_frontier(self, plot_capital_market_line = False):
        if self.fitted == False:    
            raise ValueError('Efficient frontier unknown. Please use the .fit() method.')
        
        mu_frontier = self.frontier['mu']
        std_frontier = self.frontier['std']
        mu_max_sharpe = self.portfolios['max_sharpe']['mu']
        std_max_sharpe = self.portfolios['max_sharpe']['std']
        mu_min_var = self.portfolios['min_variance']['mu']
        std_min_var = self.portfolios['min_variance']['std']

        plt.plot(std_frontier, mu_frontier)
        plt.plot(std_max_sharpe, mu_max_sharpe, color = 'r', marker = 'x', label ='Maximal Sharpe ratio')
        plt.plot(std_min_var, mu_min_var, color = 'g', marker = 'x', label = 'Minimum variance')

        if plot_capital_market_line == True:
            weights = np.linspace(0, 2, 2 * 10**3 + 1)
            mu_cml = (1 - weights) * self.r_f + weights * mu_max_sharpe 
            std_cml = weights * std_max_sharpe
            plt.plot(std_cml, mu_cml, label = 'Capital market line')

        plt.title('Portfolio frontier')
        plt.xlabel("Standard deviation")
        plt.ylabel("Expected returns")
        plt.legend()
        plt.show()


def monte_carlo_portfolios(data, number_of_simulations=10**4):
    """
    Generate random portfolios with the data given. 
    Returns a list of tuples consisting of standard deviation and expected returns of these portfolios.  
    """
    random_portfolios = []
    mu_obs = np.mean(data, axis = 0)
    cov_obs = np.cov(data, rowvar = False)
    for i in range(number_of_simulations):
        weights_mc = np.random.rand(n_stocks)
        weights_mc = weights_mc / np.sum(weights_mc)
        mu_mc = weights_mc @ mu_obs
        std_mc = np.sqrt(weights_mc @ cov_obs @ weights_mc)
        random_portfolios.append((std_mc, mu_mc))

    return random_portfolios





if __name__ == '__main__':
    np.random.seed(1)
    n_stocks = 10
    time_series_length = 10**3

    mu_intervals = sorted([[0, 1] for i in range(n_stocks)])
    std_intervals = sorted([[1,2] for i in range(n_stocks)], reverse = True)
    parameters_dict = {'mu': mu_intervals, 'sigma': std_intervals}

    dg = DataGenerator(N_dim = n_stocks)
    dg.generate_parameters(parameters_dict)
    dg.generate_correlation()

    data = dg.generate_data(N_samples = time_series_length)
    r_f = 0.2

    # Construct the efficient frontier. Plot it together with CML, optimal portfolio & min variance portfolio
    model = EfficientFrontier(data, r_f = r_f)
    model.fit()
    model.plot_efficient_frontier(plot_capital_market_line = True)

    # Use monte carlo simulations to generate random portfolios. Note that 
    # 1) Random portfolios lie within the frontier
    # 2a) As the number of stocks increases, the random portfolios converge in terms of risks / returns 
    # 2b) The convergence is not so bad in terms of risk (due to higher likelihood of diversifaction), 
    #       but gets a lot worse for expected returns (due to inefficient portfolio choices).

    random_portfolios = monte_carlo_portfolios(data)
    std, mu = zip(*random_portfolios)
    plt.scatter(std, mu, color = 'm', alpha = 0.2)
    model.plot_efficient_frontier(plot_capital_market_line = False)


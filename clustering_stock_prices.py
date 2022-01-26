import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering

from data_generator import DataGenerator



# This script demonstrates how to use clustering to identify similarly distributed time series. 
# In particular, the time series we have in mind are stock prices, which we (naively) model as 
# uncorrelated Geometric Brownian Motion.
# 
# We define a metric between the time series 
# by looking at the difference series (i.e., stock returns) and using Anderson-Darling to 
# determine the distance between two difference series. 
# 
# A clustering algorithm can then be used on the metric to determine which time series should be 
# considered to belong to the same segment (i.e., have similar distribution). 
# In particular, both KMeans and Hierarchical Clustering are considered. 
#
# It is shown that in the trivial case (no correlation, distributions fixed by hand) that 
# given the correct number of clusters, both clustering algorithms correctly identify which time series
# have identical distributions. 




def rmse(x,y):
    rmse_value = np.sqrt( np.sum((x - y)**2) )
    return  rmse_value

def anderson_darling(x,y):
    ad_value = anderson_ksamp([x,y])
    ad_value = ad_value[0]
    return ad_value

def kolmogorv_smirnov(x,y):
    ks_value = ks_2samp(x,y)
    ks_value = ks_value[0]
    return ks_value



def get_off_diagonal_indices(list_of_ts):
    """
    Given a list of time series, returns a list of all tuples (i,j) such that 
    i, j are non-identical time series.
    """
    pairs = [(i,j) for i in list_of_ts for j in list_of_ts if j != i]
    return pairs



def compute_metric_tensor(data, distance=anderson_darling):
    """ 
    Construct a metric between various time series, using a given distance function.

    Arguments
    --------
    - data: pd.DataFrame()
        columns are assumed to consist of various time series
    - distance: function
        the distance function is required to accept two array-likes of size data.shape[1] as input, 
            returning a float

    Returns
    --------
    - g_ij: pd.DataFrame of size (data.shape[0], data.shape[0])
        the (i,j) entry contains the distance between time series i and j
    """
    if distance == 'correlation':
        g_ij = data.corr()
    else:
        ts = data.columns
        g_ij = pd.DataFrame(0, index = ts, columns = ts)
        pairs = get_off_diagonal_indices(ts)
        for pair in pairs: 
            x = data[pair[0]]
            y = data[pair[1]]
            dist = distance(x,y)   
            g_ij.loc[g_ij.index == pair[0], pair[1]] = dist
            g_ij.loc[g_ij.index == pair[1], pair[0]] = dist

    return g_ij



def lookup_table(kmeans, n_clusters):
    """
    Relabel the cluster labels of a kmeans clustering by naming them in order of the total distance of points to the cluster center.
    Used in order to be able to easily compare two sets of cluster labels for different seeds
    """

    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
    lu_table = np.zeros_like(idx)
    lu_table[idx] = np.arange(n_clusters)
    renamed_labels = lu_table[kmeans.labels_]

    return renamed_labels



def to_cluster_key(n, seed=None):
    """
    Converts integer cluster label to string label as key for dictionaries.
    """
    key_name = f'cluster_{n}'
    if seed:
        key_name = f'cluster_{n}_seed_{seed}'
    return key_name



def fit_clusters(metric, n_clusters_min=1, n_clusters_max=None,  method='kmeans', n_seeds=10, make_plots=False):
    """
    Compute a dataframe with cluster labels based on a metric and a choice of clustering methodology, 
    for any number of clusters < n_clusters_max.

    Arguments
    ---------
    metric: pd.DataFrame
        Symmetric matrix g_ij, i,j running over segments
    n_clusters_max: int < len(metric)
        Maximal number of clusters to use. For all n < n_clusters_max, clustering_labels are assigned.
        If set to None, the number of segments -1 is used as a max.
    n_clusters_min: int, 0 < n_clusters_min < n_clusters_max
        Minimal number of clusters to use.
    method: str
        Determines which clustering algorithm to apply. Currently, 'kmeans' and 'hierarchical' are implemented.
    n_seeds: int
        KMeans is sensitive to choice of seed. The fit runs for n_seed different seeds, then compares whether or not 
        the fitted cluster labels are identical or not
    make_plots: bool
        Whether or not to produce plots (elbow plots for KMeans, dendrograms for Hierarchical) to visually determine
        how many clusters to use

    Returns
    ---------
    clusters_df: pd.DataFrame
        Dataframe with segments as index, number of clusters as columns, cluster labels per segment as values.
        In case different seeds lead to different results for a certain number of clusters, labels for each seed are included.
    """

    if n_clusters_max is None:
        n_clusters_max = len(metric)
    X_unnormalized = metric.to_numpy()
    X = (X_unnormalized - np.mean(X_unnormalized)) / np.std(X_unnormalized)
    clusters_df = pd.DataFrame(index = metric.index)

    if method == 'hierarchical':
        for n in range(n_clusters_min, n_clusters_max + 1):
            hierarchical = AgglomerativeClustering(n_clusters = n, affinity = 'euclidean', linkage = 'ward')
            hierarchical.fit(X)
            clusters_df[to_cluster_key(n)] = hierarchical.labels_
        if make_plots == True:
            sch.dendrogram(sch.linkage(X, method = 'ward'))

    elif method == 'kmeans':
        clusters = {}
        for seed in range(n_seeds):
            clusters[seed] = {}
            distortions = []
            for n in range(n_clusters_min, n_clusters_max + 1):
                kmeans = KMeans(n_clusters = n, random_state = seed)
                kmeans.fit(X)
                distortions.append(kmeans.inertia_)
                clusters[seed][n] = lookup_table(kmeans, n)
            if make_plots == True:
                print("Seed:", seed) 
                plt.plot(distortions)
                plt.show()
        
        # For each n, check if clusters are the same for all seeds.
        # If yes, only put the unique cluster labels in the dataframe.
        # If no, put a column of cluster labels into the dataframe for each seed.
        for n in range(n_clusters_min, n_clusters_max + 1):
            no_seed_impact = True
            for seed in range(1, n_seeds):
                if (clusters[0][n] != clusters[seed][n]).any():
                    no_seed_impact = False
                    for seed2 in range(n_seeds):
                        clusters_df[to_cluster_key(n)] = clusters[0][n]
                    break
            if no_seed_impact: 
                clusters_df[to_cluster_key(n)] = clusters[0][n]
        
    else: 
        raise ValueError(f"Unknown clustering method {method}: currently, only 'kmeans' and 'hierarchical' are implemented.")

    return clusters_df

        

def vote_for_cluster(cluster_df, n_clusters, n_seed=10):
    """
    To be used in case different seeds for K-Means lead to different cluster labels: 
    determine cluster label democratically.
    """
    df = cluster_df.copy()
    columns = [to_cluster_key(n_clusters, seed = i) for i in range(n_seed)]
    df_n = df[columns].mode(axis = 'columns')
    df = df.drop(columns = columns)
    df[to_cluster_key(n_clusters)] = df_n

    return df



if __name__ == '__main__':

    n_time_points = 252
    np.random.seed(0)

    # An arbitrary (and rather unrealistic) set of parameters defining the stock price stochastic processes
    parameter_list = [(0, 0.1), (0, 0.1), (0, 0.1), (0.1, 0.1), (0.1, 0.1), (0, 0.2), (0, 0.2), (0.2, 0.1), (0.2, 0.1), (0, 0.3)]
    dimension = len(parameter_list)
    mus, sigmas = zip(*parameter_list)
    parameters_dict = {'mu': np.array(mus), 'sigma': np.array(sigmas)}
    correlation_matrix = np.identity(dimension)
    dg = DataGenerator(n_dim = dimension,
                        distribution_type = 'lognormal', 
                        parameters_dict = parameters_dict,
                        correlation = correlation_matrix,
                        stochastic_process = True,
                        np_to_df = True)
            
    # Simulate geometric Brownian motion stock prices using the specified parameters and correlation
    stock_price_time_series = dg.generate_data(n_samples = n_time_points)
    for i in range(dimension):
        plt.plot(stock_price_time_series.iloc[:, i], label = parameter_list[i])
    plt.xlabel('Time axis')
    plt.ylabel('Stock prices')
    plt.title('Simulated stock price time series')    
    plt.legend()
    plt.show()

    stock_returns = np.log(stock_price_time_series).diff()
    
    # Compute a metric between all stock returns series, then apply clustering algortihms to determine segmentation
    metric = compute_metric_tensor(stock_returns, anderson_darling)
    df_kmeans = fit_clusters(metric, n_clusters_min = 4, n_clusters_max = 6,  method = 'kmeans')
    df_hierarchical = fit_clusters(metric, n_clusters_min = 4, n_clusters_max = 6,  method = 'hierarchical')
    print('Cluster labels:')
    print(df_kmeans)
    print(df_hierarchical)




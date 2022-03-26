import warnings
import numpy as np
import networkx as nx
from scipy.stats import logistic

from mossspider.estimators.utils import fast_exp_map


def uniform_network(n, degree, pr_w=0.35, seed=None):
    """Generates a uniform random graph for a set number of nodes (n) and specified max and min degree (degree).
    Additionally, assigns a binary baseline covariate, W, to each observation.

    Parameters
    ----------
    n : int
        Number of nodes in the generated network
    degree : list, set, array
        An array of two elements. The first element is the minimum degree and the second element is the maximum degree.
    pr_w : float, optional
        Probability of W=1. W is a binary baseline covariate assigned to each unit.
    seed : int, None, optional
        Random seed to use. Default is None.

    Returns
    -------
    networkx.Graph

    Examples
    --------
    Loading the necessary functions

    >>> from mossspider.dgm import uniform_network

    Generating the uniform network

    >>> G = uniform_network(n=500, degree=[0, 2])
    """
    rng = np.random.default_rng(seed)

    # Processing degree data
    if len(degree) > 2:
        warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                      'specified bounds are used by the bound statement. So only ' +
                      str(degree[0:2]) + ' will be used', UserWarning)
    if type(degree) is float or type(degree) is int or type(degree) is str:
        raise ValueError("degree must be a container of integers")
    elif degree[0] > degree[1]:
        raise ValueError('degree thresholds must be listed in ascending order')
    elif type(degree[0]) is str or type(degree[1]) is str:
        raise ValueError('degree must be integers')
    elif type(degree[0]) is float or type(degree[1]) is float:
        raise ValueError('degree must be integers')
    elif degree[0] < 0 or degree[1] < 0:
        raise ValueError('Both degree values must be positive values')
    else:
        # Completed all checks
        pass

    # checking if even sum for degrees, since needed
    sum = 1
    while sum % 2 != 0:                                   # Degree distribution must be even
        degree_dist = list(rng.integers(degree[0],        # ... proposed degree distribution for min degree
                                        degree[1]+1,      # ... and max degree (+1 to be inclusive)
                                        size=n))          # ... for the n units
        sum = np.sum(degree_dist)                         # ... update the sum value to see if valid

    # Generate network with proposed degree distribution
    G = nx.configuration_model(degree_dist,               # Generate network
                               seed=seed)                 # ... with seed for consistency

    # Removing multiple edges!
    G = nx.Graph(G)                                       # No multi-loops in networks we consider here

    # Removing self-loops
    G.remove_edges_from(nx.selfloop_edges(G))             # No self-loops in networks we consider here

    # Generating baseline covariate W
    w = rng.binomial(n=1, p=pr_w, size=n)                 # Generate W
    for node in G.nodes():                                # Adding W to the network node attributes
        G.nodes[node]['W'] = w[node]                      # ... via simple indexing

    # Returning the completed graph
    return G


def clustered_power_law_network(n_cluster, edges=3, pr_cluster=0.75, pr_between=0.0007, pr_w=0.35, seed=None):
    """Generate a graph with the following features: follows a power-law degree distribution, high(er) clustering
    coefficient, and an underlying community structure. This graph is created by generating a number of subgraphs with
    power-law distributions and clustering. The subgraphs are generated using
    ``networkx.powerlaw_cluster_graph(n=n_cluster[...], m=edges, p=p_cluster)``. This process is repeated for each
    element in the ``n_cluster`` argument. Then the subgraphs are then randomly connected by creating random edges
    between nodes of the subgraphs.

    Parameters
    ----------
    n_cluster : list, set, array, ndarray
        Specify the N for each subgraph in the clustered power-law network via a list. List should be positive integers
        that correspond to the N for each subgraph.
    edges : int, optional
        Number of edges to generate within each cluster. Equivalent to the ``m`` argument in
        ``networkx.powerlaw_cluster_graph``.
    pr_cluster : float, optional
        Probability of a new node forming a triad with neighbors of connected nodes
    pr_between : float, optional
        Probability of an edge between nodes of each cluster. Evaluated for all node pairs, so should be relatively
        low to keep a high community structure. Default is 0.0007.
    pr_w : float, optional
        Probability of the binary baseline covariate W for the network. Default is 0.35.
    seed : int, None, optional
        Random seed. Default is None.

    Returns
    -------
    networkx.Graph

    Examples
    --------
    Loading the necessary functions

    >>> from mossspider.dgm import clustered_power_law_network

    Generating the clustered power-law network

    >>> G = clustered_power_law_network(n_cluster=[50, 50, 50, 50])
    """
    # Prep environment
    rng = np.random.default_rng(seed)
    N = nx.Graph()

    for i in range(len(n_cluster)):
        # Generate the component / subgraph
        G = nx.powerlaw_cluster_graph(int(n_cluster[i]),
                                      m=edges,
                                      p=pr_cluster,
                                      seed=int(rng.integers(10000, 500000, size=1)[0]))

        # Re-label nodes so no corresponding overlaps between node labels
        if i == 0:
            start_label = 0
        else:
            start_label = np.sum(n_cluster[:i])
        mapping = {}
        for j in range(n_cluster[i]):
            mapping[j] = start_label + j
        H = nx.relabel_nodes(G, mapping)

        # Adding component / subgraph to overall network
        N.add_nodes_from(H.nodes)
        N.add_edges_from(H.edges)

    # Creating some random connections across groups
    for i in range(len(n_cluster)):
        # Gettings IDs for the subgraph
        first_id = int(np.sum(n_cluster[:i]))
        last_id = int(np.sum(n_cluster[:i + 1]))

        # Only adding edges to > last_id
        for j in range(first_id + 1, last_id + 1):
            for n in list(N.nodes()):
                if n > last_id:
                    if rng.uniform(0, 1) < pr_between:
                        N.add_edge(j, n)

    # Generating baseline covariate W
    w = rng.binomial(n=1, p=pr_w, size=np.sum(n_cluster))   # Generate W
    for node in N.nodes():                                  # Adding W to the network node attributes
        N.nodes[node]['W'] = w[node]                        # ... via simple indexing

    # Returning the generated network
    return N


def generate_observed(graph, seed=None):
    r"""Simulates the exposure and outcome for the uniform random graph (following mechanisms are from Sofrygin & van
    der Laan 2017).

    .. math::

        A = \text{Bernoulli}(\text{expit}(-1.2 + 1.5 W + 0.6 W^s)) \\
        Y = \text{Bernoulli}(\text{expit}(-2.5 + 0.5 A + 1.5 A^s + 1.5 W + 1.5 W^s))

    Parameters
    ----------
    graph : Graph
        Graph generated by the `uniform_network` function.
    seed : int, None, optional
        Random seed to use. Default is None.

    Returns
    -------
    Network object with node attributes

    Examples
    --------
    Loading the necessary functions

    >>> from mossspider.dgm import uniform_network, generate_observed

    Generating the uniform network

    >>> G = uniform_network(n=500, degree=[0, 2])

    Generating exposure A and outcome Y for network

    >>> H = generate_observed(graph=G)

    References
    ----------
    Sofrygin O, & van der Laan MJ. (2017). Semi-parametric estimation and inference for the mean outcome of the single
    time-point intervention in a causally connected population. *Journal of Causal Inference*, 5(1).
    """
    rng = np.random.default_rng(seed)

    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])
    adj_mat = nx.adjacency_matrix(graph)

    # Calculating map(W), generating A, and adding to network
    w_s = fast_exp_map(adj_mat,
                       w,
                       measure='sum')
    a = rng.binomial(n=1, p=logistic.cdf(-1.2 + 1.5*w + 0.6*w_s), size=n)
    for node in graph.nodes():
        graph.nodes[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = fast_exp_map(adj_mat,
                       a,
                       measure='sum')
    y = rng.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    for node in graph.nodes():
        graph.nodes[node]['Y'] = y[node]

    return graph


def generate_truth(graph, p):
    """Simulates the true conditional mean outcome for a given network, distribution of W, and policy.

    The true mean under the policy is simulated as

    .. math::

        A = Bernoulli(p) \\
        Y = Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    float

    Examples
    --------
    Loading the necessary functions

    >>> from mossspider.dgm import uniform_network, generate_truth

    Generating the uniform network

    >>> G = uniform_network(n=500, degree=[0, 2])

    Calculating truth for a policy via a large number of replicates

    >>> true_p = []
    >>> for i in range(1000):
    >>>     y_mean = generate_truth(graph=G, p=0.5)
    >>>     true_p.append(y_mean)
    >>> np.mean(true_p)  # 'true' value for the stochastic policy

    To reduce random error, a large number of replicates should be used
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.nodes[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    adj_mat = nx.adjacency_matrix(graph)
    w_s = fast_exp_map(adj_mat,
                       w,
                       measure='sum')
    a_s = fast_exp_map(adj_mat,
                       a,
                       measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    return np.mean(y)

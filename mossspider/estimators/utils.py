import warnings
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm


def probability_to_odds(prob):
    """Converts given probability (proportion) to odds

    Parameters
    ----------
    prob : float, array
        Probability or array of probabilities to convert to odds
    """
    return prob / (1 - prob)


def odds_to_probability(odds):
    """Converts given odds to probability

    Parameters
    ----------
    odds : float, array
        Odds or array of odds to convert to probabilities
    """
    return odds / (1 + odds)


def exp_map(graph, var):
    """Slow implementation of the exposure mapping functionality. Only supports the sum summary measure.
    Still used by the dgm files.

    Note
    ----
    Depreciated and no longer actively used by any functions.

    Parameters
    ----------
    graph : networkx.Graph
        Network to calculate the summary measure for.
    var : str
        Variable in the graph to calculate the summary measure for

    Returns
    -------
    array
        One dimensional array of calculated summary measure
    """
    # get adjacency matrix
    matrix = nx.adjacency_matrix(graph, weight=None)
    # get node attributes
    y_vector = np.array(list(nx.get_node_attributes(graph, name=var).values()))
    # multiply the weight matrix by node attributes
    wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
    return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...


def fast_exp_map(matrix, y_vector, measure):
    r"""Improved (computation-speed-wise) implementation of the exposure mapping functionality. Further supports a
    variety of summary measures. This is accomplished by using the adjacency matrix and vectors to efficiently
    calculate the summary measures (hence the function name). This is an improvement on previous iterations of this
    function.

    Available summary measures are

    Sum (``'sum'``) :

    .. math::

        X_i^s = \sum_{j=1}^n X_j \mathcal{G}_{ij}

    Mean (``'mean'``) :

    .. math::

        X_i^s = \sum_{j=1}^n X_j \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Variance (``'var'``):

    .. math::

        \bar{X}_j = \sum_{j=1}^n X_j \mathcal{G}_{ij} \\
        X_i^s = \sum_{j=1}^n (X_j - \bar{X}_j)^2 \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Mean distance (``'mean_dist'``) :

    .. math::

        X_i^s = \sum_{j=1}^n (X_i - X_j) \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Variance distance (``'var_dist'``) :

    .. math::

        \bar{X}_{ij} = \sum_{j=1}^n (X_i - X_j) \mathcal{G}_{ij} \\
        X_i^s = \sum_{j=1}^n ((X_j - X_j) - \bar{X}_{ij})^2 \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Note
    ----
    If you would like other summary measures to be added or made available, please reach out via GitHub.

    Parameters
    ----------
    matrix : array
        Adjacency matrix. Should be extract from a ``networkx.Graph`` via ``nx.adjacency_matrix(...)``
    y_vector : array
        Array of the variable to calculate the summary measure for. Should be in same order as ``matrix`` for
        calculation to work as intended.
    measure : str
        Summary measure to calculate. Options are provided above.

    Returns
    -------
    array
        One dimensional array of calculated summary measure
    """
    if measure.lower() == 'sum':
        # multiply the weight matrix by node attributes
        wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
        return np.asarray(wy_matrix).flatten()         # converting between arrays and matrices...
    elif measure.lower() == 'mean':
        rowsum_vector = np.sum(matrix, axis=1)         # calculate row-sum (denominator / degree)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)  # calculate each nodes weight
        wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)  # multiply matrix by node attributes
        return np.asarray(wy_matrix).flatten()         # converting between arrays and matrices...
    elif measure.lower() == 'var':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(a * y_vector, axis=1)
    elif measure.lower() == 'mean_dist':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector      # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(c.transpose(),           # back-transpose
                              axis=1)
    elif measure.lower() == 'var_dist':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector      # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(c.transpose(),            # back-transpose
                             axis=1)
    else:
        raise ValueError("The summary measure mapping" + str(measure) + "is not available")


def exp_map_individual(network, variable, max_degree):
    """Summary measure calculate for the non-parametric mapping approach described in Sofrygin & van der Laan (2017).
    This approach works best for networks with uniform degree distributions. This summary measure generates a number
    of columns (a total of ``max_degree``). Each column is then an indicator variable for each observation. To keep
    all columns the same number of dimensions, zeroes are filled in for all degrees above unit i's observed degree.

    Parameters
    ----------
    network : networkx.Graph
        The NetworkX graph object to calculate the summary measure for.
    variable : str
        Variable to calculate the summary measure for (this will always be the exposure variable internally).
    max_degree : int
        Maximum degree in the network (defines the number of columns to generate).

    Returns
    -------
    dataframe
        Data set containing all generated columns
    """
    attrs = []
    for i in network.nodes:
        j_attrs = []
        for j in network.neighbors(i):
            j_attrs.append(network.nodes[j][variable])
        attrs.append(j_attrs[:max_degree])

    return pd.DataFrame(attrs,
                        columns=[variable+'_map'+str(x+1) for x in range(max_degree)])


def network_to_df(graph):
    """Take input network and converts all node attributes to a pandas DataFrame object. This dataframe is then used
    within ``NetworkTMLE`` internally.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with node attributes to transform into data set

    Returns
    -------
    dataframe
        Data set containing all node attributes
    """
    return pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')


def bounding(ipw, bound):
    """Internal function to bound or truncate the estimated inverse probablity weights.

    Parameters
    ----------
    ipw : array
        Estimate inverse probability weights to truncate.
    bound : list, float, int, set, array
        Bounds to truncate weights by.

    Returns
    -------
    array
        Truncated inverse probability weights.
    """
    if type(bound) is float or type(bound) is int:  # Symmetric bounding
        if bound > 1:
            ipw = np.where(ipw > bound, bound, ipw)
            ipw = np.where(ipw < 1 / bound, 1 / bound, ipw)
        elif 0 < bound < 1:
            ipw = np.where(ipw < bound, bound, ipw)
            ipw = np.where(ipw > 1 / bound, 1 / bound, ipw)
        else:
            raise ValueError('Bound must be a positive value')
    elif type(bound) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float or integer, or a collection')
    else:  # Asymmetric bounds
        if bound[0] > bound[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bound) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bound[0:2]) + ' will be used', UserWarning)
        if type(bound[0]) is str or type(bound[1]) is str:
            raise ValueError('Bounds must be floats or integers')
        if bound[0] < 0 or bound[1] < 0:
            raise ValueError('Both bound values must be positive values')
        ipw = np.where(ipw < bound[0], bound[0], ipw)
        ipw = np.where(ipw > bound[1], bound[1], ipw)
    return ipw


def outcome_learner_fitting(ml_model, xdata, ydata):
    """Internal function to fit custom_models for the outcome nuisance model.

    Parameters
    ----------
    ml_model :
        Unfitted model to be fit.
    xdata : array
        Covariate data to fit the model with
    ydata : array
        Outcome data to fit the model with

    Returns
    -------
    Fitted user-specified model
    """
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    return fm


def outcome_learner_predict(ml_model_fit, xdata):
    """Internal function to take a fitted custom_model for the outcome nuisance model and generate the predictions.

    Parameters
    ----------
    ml_model_fit :
        Fitted user-specified model
    xdata : array
        Covariate data to generate the predictions with.

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    if hasattr(ml_model_fit, 'predict_proba'):
        g = ml_model_fit.predict_proba(xdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(ml_model_fit, 'predict'):
        return ml_model_fit.predict(xdata)
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def exposure_machine_learner(ml_model, xdata, ydata, pdata):
    """Internal function to fit custom_models for the exposure nuisance model and generate the predictions.

    Parameters
    ----------
    ml_model :
        Unfitted model to be fit.
    xdata : array
        Covariate data to fit the model with
    ydata : array
        Outcome data to fit the model with
    pdata : array
        Covariate data to generate the predictions with.

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # Fitting model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(pdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(fm, 'predict'):
        g = fm.predict(pdata)
        return g
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def targeting_step(y, q_init, ipw, verbose):
    r"""Estimate :math:`\eta` via the targeting model

    Parameters
    ----------
    y : array
        Observed outcome values.
    q_init : array
        Predicted outcome values under the observed values of exposure.
    ipw : array
        Estimated inverse probability weights.
    verbose : bool
        Whether to print the summary details of the targeting model.

    Returns
    -------
    float
        Estimated value to use to target the outcome model predictions
    """
    f = sm.families.family.Binomial()
    log = sm.GLM(y,  # Outcome / dependent variable
                 np.repeat(1, y.shape[0]),  # Generating intercept only model
                 offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
                 freq_weights=ipw,  # Weighted by calculated IPW
                 family=f).fit(maxiter=500)

    if verbose:  # Optional argument to print each intermediary result
        print('==============================================================================')
        print('Targeting Model')
        print(log.summary())

    return log.params[0]  # Returns single-step estimated Epsilon term


def tmle_unit_bounds(y, mini, maxi):
    """Bounding for continuous outcomes for TMLE.

    Parameters
    ----------
    y : array
        Observed outcome values
    mini : float
        Lower bound to apply
    maxi : float
        Upper bound to apply

    Returns
    -------
    array
        Bounded outcomes
    """
    return (y - mini) / (maxi - mini)


def tmle_unit_unbound(ystar, mini, maxi):
    """Unbound the bounded continuous outcomes for presentation of results.

    Parameters
    ----------
    ystar : array
        Bounded outcome values
    mini : float
        Lower bound to apply
    maxi : float
        Upper bound to apply

    Returns
    -------
    array
        Unbounded outcomes
    """
    return ystar*(maxi - mini) + mini


def create_threshold(data, variables, thresholds):
    """Internal function to create threshold variables given setup information.

    Parameters
    ----------
    data : dataframe
        Data set to calculate the measure for
    variables : list, set
        List of variable names to create the threshold variables for
    thresholds : list, set
        List of values (float or int) to create the thresholds at.

    Returns
    -------
    None
    """
    for v, t in zip(variables, thresholds):
        if type(t) is float:
            label = v + '_t' + str(int(t * 100))
        else:
            label = v + '_t' + str(t)
        data[label] = np.where(data[v] > t, 1, 0)


def create_categorical(data, variables, bins, labels, verbose=False):
    """

    Parameters
    ----------
    data : dataframe
        Data set to calculate the measure for
    variables : list, set
        List of variable names to create the threshold variables for
    bins : list, set
        List of lists of values (float or int) to create bins at.
    labels : list, set
        List of lists of labels (str) to apply as the new column names
    verbose : bool, optional
        Whether to warn the user if any NaN values occur (a result of bad or incompletely specified bins). Interally,
        this option is always set to be True (since important for user to recognize this issue).

    Returns
    -------
    None
    """
    for v, b, l in zip(variables, bins, labels):
        col_label = v + '_c'
        data[col_label] = pd.cut(data[v],
                                 bins=b,
                                 labels=l,
                                 include_lowest=True).astype(float)
        if verbose:
            if np.any(data[col_label].isna()):
                warnings.warn("It looks like some of your categories have missing values when being generated on the "
                              "input data. Please check pandas.cut to make sure the `bins` and `labels` arguments are "
                              "being used correctly.", UserWarning)

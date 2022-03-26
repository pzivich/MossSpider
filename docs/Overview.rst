Overview
=====================================

Here, we will provide an example application of ``mossspider`` to highlight some of the available features.

Data Generation
-------------------------------------
Before demonstrating the application of network-TMLE, we use ``mossspider`` to generate some generic example data.
Here, we will generate both a network and the covariates for that network.

Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``mossspider`` provides a few functions to randomly generate networks with different structural features. Here, we
will use the ``mossspider.dgm.uniform_network`` function. We will generate a network with a uniform degree distribution
with degrees between 1-4 and consists of 500 nodes.

.. code::

    # importing uniform network
    from mossspider.dgm import uniform_network

    G = uniform_network(n=500,          # Number of nodes
                        degree=[1, 4],  # Min and Max degree
                        seed=2022)      # Seed for consistency

The network generation functions further assign baseline covariates ``W`` in the network. For the estimand
described, ``W`` is assumed to be held constant in the super-population of networks. Therefore, the data generation step
only assigns baseline covariates once. Here, ``W`` consists of a single binary covariate.

Truth or Reference Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we will generate data. First, we can use the ``mossspider.dgm.generate_truth`` function to estimate the mean
under the policy of interest, :math:`\omega`. This function takes the specified policy, applies it to the network,
calculates the outcomes from the true outcome model, and then returns the mean. To estimate the truth for the
super-population of networks, we run this function a 'large' number of times and take the mean of the means. Below is
code that does this for a policy where everyone has their probability of action :math:`A` set to 0.65.

.. code::

    import numpy as np
    from mossspider.dgm import generate_truth

    # Setup values to evaluate at
    omega = 0.65                          # Policy of interest
    true_p = []                           # Empty storage

    # Calculate truth or reference values
    for i in range(5000):                 # Sim 5k times
        y_mean = generate_truth(graph=G,  # Mean for graph
                                p=omega)  # ... under omega
        true_p.append(y_mean)             # Store mean

    truth = np.mean(true_p)               # Calculate mean of means
    print(truth)

Therefore, we have simulated what the estimand is expected to be. Remember, that this estimand will change based on
the distribution of :math:`\mathbf{W}`. Therefore, changing the seed in the generation of ``G`` will result in a
different truth value here.

Observed Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we can simulate the observed data. Instead of using the policy of interest, :math:`A` and :math:`Y` are assigned
according to some mechanism that is not the policy of interest. In practice, this mechanism is unknown and consists of
the nuisance models that must be estimated to use network-TMLE. We will do this using the
``mossspider.dgm.generate_observed`` function, which returns a network with assigned actions and outcomes

.. code::

    from mossspider.dgm import generate_observed

    H = generate_observed(G, seed=202203)

Notice that if you examine the network, nodes have three attributes: ``W``, ``A``, and ``Y``. ``NetworkTMLE`` expects
the input data to be formatted in a similar manner (a ``networkx.Graph`` object with assigned node attributes).

Network-TMLE
-------------------------------------
``mossspider`` implements network-TMLE through the ``NetworkTMLE`` function. The following details how ``NetworkTMLE``
operates and broadly what happens behind the scenes.

Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As mentioned, ``NetworkTMLE`` expects the data to be provided in a particular form. This is to ensure all the
calculations and data extractions go smoothly behind the scenes. Most importantly, ``NetworkTMLE`` expects the data to
be provided as a ``networkx.Graph`` object. Furthermore, all covariates must be provided as node attributes.

Below is the initialization of ``NetworkTMLE`` for the previously generated data set

.. code::

    ntmle = NetworkTMLE(network=H,     # NetworkX graph
                        exposure='A',  # Exposure in graph
                        outcome='Y',   # Outcome in graph
                        verbose=True)  # Print model summaries

Besides the network, ``NetworkTMLE`` requires that the label for the action (referred to as exposure here) nad the label
for the outcome in the graph are provided. There are optional arguments for the confidence-level (``alpha``), whether
to apply a restriction based on degree (``degree_restrict``), and whether to display nuisance model summary information
(``verbose``). By default no degree restriction is applied and 95% confidence intervals are provided.

Behind the scenes, ``NetworkTMLE`` extracts the covariates from the graph, creates a ``pandas.DataFrame``, and
calculates summary measures. Covariates are provided via ``networkx.Graph`` instead of a ``pandas.DataFrame`` to ensure
that summary measures are all correctly calculated. This is then merged with the degree of each node (with the optional
degree restriction applied). Finally, storage for intermediate pieces are created. Continuous outcomes are further
bounded to be :math:`(0,1)` for the targeting step later on.

Exposure Nuisance Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we need to specify the exposure nuisance model. These models are used to calculate the following weights:

.. math::

    \frac{\Pr^*(A,A^s | W, W^s)}{\Pr(A,A^s | W, W^s)}

where the numerator is from the policy of interest and the denominator is based on the observed distribution of actions.
Here, we estimate these models by factoring the probabilities as

.. math::

    \Pr(A,A^s | W, W^s) = \Pr(A | W, W^s) \Pr(A^s | A, W, W^s)

Therefore, two models need to be specified: one for :math:`A`, and one for :math:`A^s`. For :math:`A`, we will use a
logistic model

.. code::

    # Model for Pr(A | W, W^s)
    ntmle.exposure_model(model="W + W_sum",  # Parametric model
                         custom_model=None)  # ... optional argument

Certain flexible models (e.g, sci-kit learn models) can also be used. Note that these must be classifiers and are
provided via the optional ``custom_model`` argument.

Next, a model for the summary measure needs to be specified. Importantly, the summary measure and an appropriate model
must be selected. For available summary measures, see the Summary Measures page. Here, we will use the following
summary measure

.. math::

    A_i^s = \sum_{j=1}^{n} A_j \mathcal{G}_{ij}

where :math:`\mathcal{G}` is the adjacency matrix. This summary measure is a simple count of the immediate contacts with
:math:`A=1`. Now, we can specify the exposure mapping model

.. code::

    # Model for Pr(A^s | A, W, W^s)
    ntmle.exposure_map_model(model='A + W + W_sum',  # Parametric model
                             measure='sum',          # Summary measure for A^s
                             distribution='poisson') # Model distribution to use

Here, the model must be provided as well as the summary measure (``measure``) and the distribution to use for the model
(``distribution``). Since our summary measure is a count, we use a Poisson regression model. While ``custom_models``
are provided, care must be taken to ensure that the distribution of that custom model agrees with the ``distribution``
argument. Otherwise, weights **will not** be estimated correctly.

In both of these steps, we are only specifying the parametric form of these models and the summary measures to use. The
actual estimation of the weights is done later in the ``NetworkTMLE.fit`` step.

Outcome Nuisance Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we need to specify and estimate the outcome nuisance model: :math:`E[Y | A, A^s, W, W^s]`. Unlike the weights,
we can (and will) estimate the outcome model in this function. To specify the outcome model

.. code::

    # Model for E[Y | A, A^s, W, W^s]
    ntmle.outcome_model(model='A + A_sum + W + W_sum',
                        custom_model=None)

For binary outcomes (internally detected in the initialization), a logistic model is used. For continuous outcomes, the
default is linear regression but other models can be used by specifying the optional ``distribution`` argument. Finally,
custom models can also be used here. There is more flexibility in what algorithms could be considered (since we only
need the predicted values).

Notice that the summary measure for the outcome nuisance model and the exposure nuisance model are the same for
:math:`A^s`.

Behind the scenes, the function saves the model specification, fits the specified outcome model, and generates predicted
values of the outcome under the observed values of :math:`A` and :math:`A^s`. These estimates are all stored interally
for the next step.

Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, we can estimate the conditional mean under the policy of interest. ``NetworkTMLE`` takes the policy in the
form of a float (which sets everyone to the same probability of having :math:`A=1`) or as a vector (assigns each unit
their own probability of :math:`A=1`). Here, the policy of interest is :math:`\Pr(A_i)=0.65`.

.. code::

    # Estimation
    ntmle.fit(p=0.65,          # Policy
              samples=500,     # ... replicates for MC integration
              bound=None,      # ... option to bound weights
              seed=20220316)   # ... seed for consistency

Other optional arguments include settings the number of samples to use in the Monte Carlo integration procedure
(``samples``, see below for details on this), truncation of estimated weights (``bound``), and a random seed for
consistent results of the estimation procedure.

Behind the scenes, there are lots of steps that occur. First, checks are applied to make sure the nuisance models are
all specified and the policy is been specified in a compatible format. Next, the weights are estimated. This is done
by estimating the denominator using the observed data. For the numerator, we can't use the policy of interest directly
(since it is specified in terms of :math:`A_i` and not :math:`A_i,A_i^s`). Therefore, we use a Monte-Carlo procedure.
Briefly, we generate ``samples`` copy of the data. To each copy, the stochastic policy is applied. Using all copies of
the data with the copy of the stochastic policy applied simultaneously, the exposure nuisance models are estimated.
Then the observed :math:`A_i,A_i^s` and estimated model parameters are used to estimate the numerator. If ``bound`` is
specified, the weights are then bounded.

Next, the targeting step is applied. This involves taking the predicted values from ``NetworkTMLE.outcome_model`` and
the estimated weights and fitting a weighted intercept-only logistic model. Then the outcome model is used to predict
the outcome under the policy of interest and is updated using the estimated targeting model. Since stochastic policies
have a number of different possible distributions, a Monte-Carlo procedure is again used. Here, we re-use the data sets
generated in the weight estimation step. Using the :math:`A_i,A_i^s` under the policy, predicted values of the outcomes
are generated, updated via the targeting model, averaged over each data set, and finally averaged across the
``samples``.

Finally, the variance is calculated. Two variances are calculated. The first assumes that all dependence is due to
direct transmission only, while the second allows for direct and latent transmission. For theoretical reasons, the
latter will generally be preferred.

Note that increasing ``samples`` will result in a more 'stable' estimate (it will be less subject to random noise if a
different seed had been used). Personally, I have found good performance with 100-500. Ideally, you would run as much
as possible. Unfortunately, the most computationally intensive part is the generation of copies of the data set.
Therefore, run-times are highly dependent on the value used for ``samples``.

Summary Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A summary of the results can be printed to the console via:

.. code::

    # Displaying results
    ntmle.summary(decimal=4)

To increase the number of decimals displayed, use the ``decimal`` argument.

Diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, we have a diagnostic available. The diagnostic provides a plot to visually assess how well-supported the policy
of interest is by the observed distribution of :math:`\mathbf{A}`. Briefly, the diagnostic plots the summary measure
:math:`A_i^s` by :math:`A_i` in the observed data. This is then contrasted with :math:`A_i^s` under the policy (as
generated in the Monte-Carlo step). For well-supported policies, the observed data and generated data under the policy
should overlap. If there is little overlap, this is indicative of the policy of interest being poorly-supported by the
data. Poorly-supported policies can result in biased estimation and poor confidence interval coverage. For details
see [...].

The diagnostic plot can be generated via

.. code::

    import matplotlib.pyplot as plt

    ntmle.diagnostics()
    plt.show()

Additional Examples
-------------------------------
Additional examples are provided `here<https://github.com/pzivich/MossSpider/tree/main/examples>`_.

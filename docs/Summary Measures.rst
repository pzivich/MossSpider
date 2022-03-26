Summary Measures
==============================

The following provides documentation for the available summary measures in ``mossspider``. Currently, ``mossspider``
does not support fully custom summary measures. We are working on how to best implement this option.

Column names for all summary measures currently in the data can be checked via

.. code::

    ntmle = NetworkTMLE(G, exposure='A', outcome='Y')
    print(ntmle.df.columns)

Basic Measures
------------------------------
The following basic summary measures are always available for model specifications. They are all calculated by default.
They include: sum, mean, variance, mean distance, and variance distance.

Throughout this section, let :math:`X` indicate the covariate the summary measure is being calculated for, and
:math:`\mathcal{G}` indicate the adjacency matrix for the network.

Sum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The sum summary measure is defined as

.. math::

    X_i^s = \sum_{i=1}^{n} X_j \mathcal{G}_{ij}

For the covariate ``X`` in the data, the sum summary measure column is accessed by ``X_sum``.

Mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The mean summary measure is defined as

.. math::

    X_i^s =  \frac{\sum_{i=1}^{n} X_j \mathcal{G}_{ij}}{\sum_{i=1}^{n} \mathcal{G}_{ij}}

For the covariate ``X`` in the data, the mean summary measure column is accessed by ``X_mean``.

Variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The variance summary measure is defined as

.. math::

    X_i^s =  \frac{\sum_{i=1}^{n} (X_j - X_i)^2 \mathcal{G}_{ij}}{\sum_{i=1}^{n} \mathcal{G}_{ij}}

For the covariate ``X`` in the data, the variance summary measure column is accessed by ``X_var``.

Mean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The mean distance summary measure is defined as

.. math::

    X_i^s =  \frac{\sum_{i=1}^{n} (X_j - X_i) \mathcal{G}_{ij}}{\sum_{i=1}^{n} \mathcal{G}_{ij}}

For the covariate ``X`` in the data, the mean distance summary measure column is accessed by ``X_mean_dist``.

Variance Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The variance distiance summary measure is defined as

.. math::

    X_i^s =  \frac{\sum_{i=1}^{n} ((X_j - X_i) - \bar{X}_i)^2 \mathcal{G}_{ij}}{\sum_{i=1}^{n} \mathcal{G}_{ij}}

where

.. math::

    \bar{X}_i = \frac{\sum_{i=1}^{n} X_j \mathcal{G}_{ij}}{\sum_{i=1}^{n} \mathcal{G}_{ij}}

For the covariate ``X`` in the data, the variance distance summary measure column is accessed by ``X_var_dist``.

Partially Custom Measures
------------------------------
Partially custom measures allow for some flexibility. These measures include a threshold measure and a category measure.
By default, these are not automatically calculated. They must be specified using the corresponding functions.

These measures are further build upon the basic measures. Therefore, familiarize yourself with the basic measures before
this section.

Threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a specified variable and summary measure, a threshold indicator variable can be created. For example, we may
want to create a summary measure of the action which is an indicator if a unit has more than 3 immediate contacts with
:math:`A=1`. To create this measure, we call

.. code::

    ntmle.define_threshold(variable='A_sum',   # Variable to use
                           threshold=3)        # ... set threshold (>, or <=)

This function should be called prior to estimating the nuisance models. Furthermore, the function calculates the
threshold measure for the observed data and the Monte-Carlo generated data automatically.

Thresholds can be created for multiple variables by specifying the ``define_threshold`` argument multiple times.

To access the threshold summary measure column, use ``'A_sum_t3'`` here. The naming convention works like the following:
variable + underscore + t + threshold.

Category
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a specified variable and summary measure, a categorization is created. For example, we may want to bin ``W_sum`` to
reduce the dimension for a power-law network while still trying to model ``W_sum`` flexibly. To create a category
summary measure based on user-specified bins, the following function is used:

.. code::

    ntmle.define_category(variable='W_sum',       # Variable to bin
                          bins=[0, 1, 3, 7, 12],  # ... bins (includes right)
                          labels=False)           # ... allow for new labels (not recommended)

From this function, a new column consisting of a categorical dummy variable is generated. The naming convention for this
new column is the variable name + underscore + c. Therefore, the new categorical variable would be ``'W_sum_c'``.

As with the threshold, this function should be called prior to estimating the nuisance models. Furthermore, the
function calculates the threshold measure for the observed data and the Monte-Carlo generated data automatically.
Finally, categories can be created for multiple variables by specifying the ``define_category`` argument multiple
times.

Fully Custom Measures
------------------------------
Not available yet.

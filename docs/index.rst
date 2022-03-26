.. image:: images/mossspider_header.png

MossSpider
=====================================

``mossspider`` provides an implementation of a targeted maximum likelihood estimator (TMLE) for network-dependent data
with stochastic policies (network-TMLE) in Python 3.6+. For in-depth details on network-TMLE, see van der Laan (2014),
Sofrygin and van der Laan (2017), or Ogburn et al. (2017). ``mossspider`` get its name from the
`spruce-fir moss spider <https://en.wikipedia.org/wiki/Spruce-fir_moss_spider>`_, a tarantula that is both the world's
smallest tarantula and native to North Carolina.

Network-TMLE is an estimator for causal effects with network-dependent data. Network-TMLE here further relies on a
weak dependence assumption (only the immediate contacts of a unit have an effect on that unit's outcome) to make
progress in this setting. This is further accomplished via parametric summary measures of immediate contacts'
covariates. The following is a brief overview. For further details, please see the references below.

Here, the estimand is the expected mean of an outcome under a policy (indicated by :math:`\omega`) for a
super-population of networks each consisting of :math:`n` units. Due to assumptions for the variance, ``mossspider``
focuses on this estimand, but is further conditional on the distribution of :math:`\mathbf{W}` covariates in the
network. This estimand can be written as

.. math::

    \psi = \frac{1}{n} \sum_{i=1}^{n} E \left[ \sum_{a\in\mathcal{A}, a^s\in\mathcal{A}^s Y_i(a,a^s) \Pr^*(A_i=a, A_i^s=a^s | W_i, W_i^s) | \mathbf{W} \right]

where :math:`A_i` is the action of interest for unit :math:`i` (with :math:`\mathcal{A}` indicating the support),
:math:`A_i^s` is a generic summary measure of the actions of :math:`i`'s immediate contacts ((with :math:`\mathcal{A}^s`
indicating the support)), :math:`Y(a,a^s)` is unit :math:`i`'s potential outcome under action :math:`a` and actions
:math:`a^s` of their contacts, and :math:`\mathbf{W} = \{W_1, W_2, ..., W_i, ..., W_n\}`. The values of :math:`a` and
:math:`a^s` are assigned according to the policy :math:`\omega` indicated by
:math:`\Pr^*(A_i=a, A_i^s=a^s | W_i, W_i^s)`.

For identification of :math:`\psi`, we rely on the following assumptions: causal consistency, exchangeability, and
positivity. Respectively, these are written as

.. math::

    \text{If } A_i=a, A_i^s=a^s \text{ then } Y_i = Y_i(a,a^s) \\
    Y(a,a^s) \amalg A,A^s | W,W^s \text{ for all } a \in \mathcal{A}, a^s \in \mathcal{A}^s \\
    \text{If } \Pr^*(A=a,A^s=a^s | W,W^s) > 0 \text{ then } \Pr(A=a,A^s=a^s | W,W^s) > 0 \text{ for all } a,a^s

These assumptions further require that (1) the network is perfectly measured, and (2) the parametric from of the summary
measure :math:`A^s` is known. This set of assumptions is unverifiable and thus needs to be based on substantive
knowledge.

Given these assumptions, :math:`\psi` is identified and we can estimate using network-TMLE. For how network-TMLE
operates (as implemented in ``mossspider``) see the Overview page on the sidebar.

Installation:
-------------

``mossspider`` can be downloaded using PyPI. To install ``mossspider``, use the following command in terminal or
command prompt

``python -m pip install mossspider``

There are several dependencies for ``mossspider``. These consist of NumPu, SciPy, Pandas, NetworkX, statsmodels, patsy,
and Matplotlib. *Note that NetworkX must be at least version 2.0.0 to operate properly*.

To replicate the tests in ``tests/`` you will need to install ``pytest`` (but this is not necessary for general use of
the package).

Contents:
-------------------------------------

.. toctree::
  :maxdepth: 3

  Overview.rst
  Summary Measures <Summary Measures.rst>
  Reference/index
  Create a GitHub Issue <https://github.com/pzivich/MossSpider/issues>


Code and Issue Tracker
-----------------------------

Please report bugs, issues, or feature requests on GitHub
at `pzivich/MossSpider <https://github.com/pzivich/MossSpider/>`_.

Otherwise, you may contact us via email (gmail: zivich.5) or on Twitter (@PausalZ)

References
-----------------------------
Ogburn EL, Sofrygin O, Diaz I, & van der Laan, MJ. (2017). Causal inference for social network data.
*arXiv preprint arXiv:1705.08527*.

Sofrygin O, & van der Laan MJ. (2017). Semi-parametric estimation and inference for the mean outcome of the single
time-point intervention in a causally connected population. *Journal of Causal Inference*, 5(1).

van der Laan MJ. (2014). Causal inference for a population of causally connected units. *Journal of Causal Inference*,
2(1), 13-74.

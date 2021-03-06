![mossspider](docs/images/mossspider_header.png)

![tests](https://github.com/pzivich/MossSpider/actions/workflows/python-package.yml/badge.svg)
[![version](https://badge.fury.io/py/mossspider.svg)](https://badge.fury.io/py/mossspider)
[![docs](https://readthedocs.org/projects/mossspider/badge/?version=latest)](https://mossspider.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/mossspider/month)](https://pepy.tech/project/mossspider)

# MossSpider

MossSpider provides an implementation of the targeted maximum likelihood estimator for network-dependent data
(network-TMLE) in Python. Currently `mossspider` supports estimation of the conditional network mean for stochastic
policies.

`mossspider` get its name from the [spruce-fir moss spider](https://en.wikipedia.org/wiki/Spruce-fir_moss_spider), a
tarantula that is both the world's smallest tarantula and native to North Carolina.

## Installation

### Installing:

You can install via `python -m pip install mossspider`

### Dependencies:

The dependencies are: `numpy`, `scipy`, `statsmodels`, `networkx`, `matplotlib`. Notice that NetworkX must be at least
2.0.0 to work properly.

## Getting started

To demonstrate `mossspider`, below is a simple demonstration of calculating the mean for the following data.

```python
from mossspider import NetworkTMLE
from mossspider.dgm import uniform_network, generate_observed
```

First, we will use some built-in data generating functions
```python
graph = uniform_network(n=500, degree=[1, 4])
graph_observed = generate_observed(graph)
```

Now, we can use `NetworkTMLE` to estimate the causal conditional mean under a stochastic policy. Here, the stochastic
policy sets everyone's probability of action `A=1` to 0.65.

```python
ntmle = NetworkTMLE(network=graph_observed,
                    exposure='A',  # Exposure in graph
                    outcome='Y',   # Outcome in graph
                    verbose=True)  # Print model summaries
ntmle.exposure_model(model="W + W_sum")
ntmle.exposure_map_model(model='A + W + W_sum',  # Parametric model
                         measure='sum',          # Summary measure for A^s
                         distribution='poisson') # Model distribution to use
ntmle.outcome_model(model='A + A_sum + W + W_sum')
ntmle.fit(p=0.65, samples=500)
ntmle.summary()
```

For full details on using `mossspider`, see the full documentation and worked examples available
at [MossSpider website](https://mossspider.readthedocs.io/en/latest/).

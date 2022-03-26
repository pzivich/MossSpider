import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time

from mossspider import NetworkTMLE
from mossspider.dgm import uniform_network, generate_observed, clustered_power_law_network

G = clustered_power_law_network(n_cluster=[50, 50, 50, 50], seed=8591)
print(G.nodes(data=True))

G = uniform_network(n=500, degree=[1, 4], seed=2022)

# Example with one particular data set
H = generate_observed(graph=G, seed=941012)

start = time.time()
ntmle = NetworkTMLE(network=H,
                    exposure='A',
                    outcome='Y',
                    degree_restrict=None,)
ntmle.exposure_model(model="W + W_sum")
ntmle.exposure_map_model(model='A + W + W_sum', measure='sum', distribution='poisson')
ntmle.outcome_model(model='A + A_sum + W + W_sum')
ntmle.fit(p=0.15, samples=100, seed=101)
print('time:', time.time() - start)

ntmle.summary()

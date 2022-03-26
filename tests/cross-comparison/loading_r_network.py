import numpy as np
import pandas as pd
import pandas.testing as pdt
import networkx as nx

from amonhen import NetworkTMLE


df = pd.read_csv("tmlenet_r_data.csv")
df['IDs'] = df['IDs'].str[1:].astype(int)
df['NETID_split'] = df['Net_str'].str.split()

G = nx.DiGraph()
G.add_nodes_from(df['IDs'])

# Adding edges
for i, c in zip(df['IDs'], df['NETID_split']):
    if type(c) is list:
        for j in c:
            G.add_edge(i, int(j[1:]))

# Adding attributes
for node in G.nodes():
    G.nodes[node]['W'] = np.int(df.loc[df['IDs'] == node, 'W1'])
    G.nodes[node]['A'] = np.int(df.loc[df['IDs'] == node, 'A'])
    G.nodes[node]['Y'] = np.int(df.loc[df['IDs'] == node, 'Y'])


tmle = NetworkTMLE(network=G, exposure='A', outcome='Y',  verbose=False)
# print(tmle.df[['_original_id_', "A_map1", "A_map2"]].head(30))
tmle.df.to_csv("tmlenet_r_processed.csv")
tmle.exposure_model('W + W_sum')
tmle.exposure_map_model('A + W + W_sum', measure=None, distribution=None)
tmle.outcome_model('A + A_sum + W + W_sum')
for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    tmle.fit(p=prob, samples=1000, bound=0.005)
    tmle.summary(decimal=6)


# tmle = NetworkTMLE(network=G, exposure='A', outcome='Y',  verbose=False)
# tmle.exposure_model('W + W_sum')
# tmle.exposure_map_model('A + W + W_sum', measure="sum", distribution="poisson")
# tmle.outcome_model('A + A_sum + W + W_sum')
# tmle.fit(p=0.65, samples=100, bound=0.005)
# tmle.summary(decimal=6)

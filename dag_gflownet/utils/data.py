import pandas as pd
import numpy as np
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.sampling import GibbsSampling
from pgmpy.models import MarkovNetwork
import networkx as nx

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'

    elif name =='random_latent_graph':
        graph, data = get_random_graph(d=args.x_dim, D=ags.h_dim, n=args.num_samples)
        score = None
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score


def get_random_graph(d, D, n):
    latent_nodes = ['h' + str(i) for i in range(d)]
    obs_nodes = ['x' + str(i) for i in range(D)]
    # Random Graph
    edges = []
    is_edge_list = np.random.binomial(1, 0.6, 2 ** d)

    for e0_idx in range(d):
        for e1_idx in range(d):
            if is_edge_list[e0_idx * d + e1_idx] and e0_idx != e1_idx:
                edges.append((latent_nodes[e0_idx], latent_nodes[e1_idx]))

    # Adding edges between latent_nodes and obs_nodes
    for l in latent_nodes:
        edges += [(l, obs) for obs in obs_nodes]

    # Adding edges among obs_nodes themselevs
    for i in range(len(obs_nodes)):
        for j in range(i + 1, len(obs_nodes)):
            edges += [(obs_nodes[i], obs_nodes[j])]

    model = MarkovNetwork(edges)
    cliques = list(map(tuple, nx.find_cliques(model.triangulate())))
    factors_list = [DiscreteFactor(list(clique), [2] * len(list(clique)),
                                   np.random.rand(2 ** (len(list(clique))))) for clique in cliques]

    model.add_factors(*factors_list)
    gibbs = GibbsSampling(model)
    data = gibbs.sample(size=n)

    return model, data

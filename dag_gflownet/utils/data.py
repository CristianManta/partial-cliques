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

def get_clique_selection_mask(gfn_state : tuple,
                              unobserved_cliques : list,
                              K : int):
    '''
    Given a GFN state and a mutable representation of unfinished cliques,
    return a mask of eligible variables for the clique selection policy.

    Inputs
    --------
    gfn_state : tuple
        There are three iterables of the same length (N) in this tuple.
        The first iterable is binary and denotes observed variables.
        The second iteration can take on K+1 values and denote the 
        if a value has been sampled for each observed variable, and 
        if so, what that value is.
        The third iterable is binary and denotes if a variable has
        never been cashed out as a part of a reward term.
    
    unobserved_cliques : list
        A list of sets, where each set correspond to a clique. Each 
        variable is represented by its index, an integer, Fully observed 
        and cashed out variables are excluded from these sets.
    
    K : int
        The number of possible values.

    Outputs
    --------
    mask : list
        A binary list of size N denoting the eligibility of each variable to be
        selected by the clique selection policy.
    '''
    assert len(gfn_state) == 3
    assert len(gfn_state[0]) == len(gfn_state[1])
    assert len(gfn_state[0]) == len(gfn_state[2])
    assert len(np.unique(gfn_state[0])) <= 2
    assert len(np.unique(gfn_state[2])) <= 2
    assert np.max(gfn_state[1]) <= K
    
    N = len(gfn_state[0])
    active_vars = set(np.nonzero(gfn_state[2]&gfn_state[0])[0].flatten())
    eligible_cliques = list(filter(lambda c:c.issuperset(active_vars), unobserved_cliques))
    eligible_vars = set().union(*eligible_cliques) - set(active_vars)
    
    mask = np.zeros(N)
    mask[np.array(list(eligible_vars))] = 1
    return mask


def get_value_policy_reward(gfn_state : tuple,
                            unobserved_cliques : list,
                            clique_potentials : list,
                            K : int):
    '''
    Given a GFN state and a mutable representation of unfinished cliques,
    return a mask of eligible variables for the clique selection policy.

    Inputs
    --------
    gfn_state : tuple
        There are three iterables of the same length (N) in this tuple.
        The first iterable is binary and denotes observed variables.
        The second iteration can take on K+1 values and denote the 
        if a value has been sampled for each observed variable, and 
        if so, what that value is.
        The third iterable is binary and denotes if a variable has
        never been cashed out as a part of a reward term.
    
    unobserved_cliques : list
        A list of sets, where each set correspond to a clique. Each 
        variable is represented by its index, an integer, Fully observed 
        and cashed out variables are excluded from these sets.
    
    clique_potentials : list
        A list of potential functions, each corresponding to a clique.
    
    K : int
        The number of possible values.

    Outputs
    --------
    new_gfn_state : tuple
        After marking the new clique as having been cashed out.
    new_unobserved_cliques : list
        After removing nodes that are fully observed
    reward : float
        The energy term associated with the clique that has been cashed out.
    '''
    assert len(gfn_state) == 3
    assert len(gfn_state[0]) == len(gfn_state[1])
    assert len(gfn_state[0]) == len(gfn_state[2])
    assert len(np.unique(gfn_state[0])) <= 2
    assert len(np.unique(gfn_state[2])) <= 2
    assert len(unobserved_cliques) == len(clique_potentials)
    assert np.max(gfn_state[1]) <= K
    
    # we remove fully observed nodes
    newly_observed_vars = set(np.nonzero(gfn_state[0]&gfn_state[2])[0].flatten())
    new_unobserved_cliques = [c - newly_observed_vars for c in unobserved_cliques]

    # we cash in every clique we complete and update the GFN state
    num_cliques = len(unobserved_cliques)
    reward = 0.
    for c_ind in range(num_cliques):
        if len(new_unobserved_cliques[c_ind]) == 0 and \
           len(unobserved_cliques[c_ind]) != 0:
            gfn_state[2][np.array(list(unobserved_cliques[c_ind]))] = 0
            reward += \
                clique_potentials[c_ind](gfn_state[1][np.array(sorted(list(unobserved_cliques[c_ind])))])

    return gfn_state, new_unobserved_cliques, reward


if __name__ == "__main__":
    
    
    seed = 0
    np.random.seed(seed)
    
    """
    Testing get_clique_selection_mask
    """    
    K = 2
    # Setting up a dummy GFN state
    # Assuming that we have 10 variables, x_0^3 and h_0^5
    # We have two cliques {x_0^3, h_0^2} and {x_0^3, h_3^5}
    # We have only fully observed x_0^3 and h_0
    gfn_state = (np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
                 np.array([1, 2, 2, 2, 2, 2, 0, 1, 0, 1]),
                 np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]),
                          set([3, 4, 5, 6, 7, 8, 9])]

    mask = get_clique_selection_mask(gfn_state, unobserved_cliques, K)
    assert np.all(mask == np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])), mask

    # Setting up a list of dummy potential functions
    clique_potentials = [lambda c : 0.2,
                         lambda c : -2]
    gfn_state = (np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
                 np.array([1, 1, 0, 2, 2, 2, 0, 1, 0, 1]),
                 np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]),
                          set([3, 4, 5, 6, 7, 8, 9])]
    new_gfn_state, new_unobserved_cliques, reward = \
        get_value_policy_reward(gfn_state, unobserved_cliques, clique_potentials, K)
    assert reward == 0.2
    assert len(new_unobserved_cliques[0]) == 0
    gfn_state = (np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1]),
                 np.array([1, 1, 0, 2, 0, 2, 0, 1, 0, 1]),
                 np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]))
    new_gfn_state, new_unobserved_cliques, reward = \
        get_value_policy_reward(new_gfn_state, new_unobserved_cliques, clique_potentials, K)
    assert reward == 0.

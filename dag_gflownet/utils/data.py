import pandas as pd
import numpy as np
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_sum_product
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

    with open(filename, "wb") as f:
        f.write(file_content)

    return filename


def get_data(name, args, rng=default_rng()):
    if name == "erdos_renyi_lingauss":
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng,
        )
        data = sample_from_linear_gaussian(graph, num_samples=args.num_samples, rng=rng)
        score = "bge"

    elif name == "sachs_continuous":
        graph = get_example_model("sachs")
        filename = download(
            "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz",
            Path("data/sachs.data.txt"),
        )
        data = pd.read_csv(filename, delimiter="\t", dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = "bge"

    elif name == "sachs_interventional":
        graph = get_example_model("sachs")
        filename = download(
            "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz",
            Path("data/sachs.interventional.txt"),
        )
        data = pd.read_csv(filename, delimiter=" ", dtype="category")
        score = "bde"

    elif name == "random_latent_graph":
        graph, (cliques, factors), data = get_random_graph(
            d=args.h_dim,
            D=args.x_dim,
            n=args.num_samples,
            m=args.num_eval_samples,
            rng=rng,
            latent_structure=args.latent_structure,
        )
        graph = (graph, cliques, factors)
        score = None
    else:
        raise ValueError(f"Unknown graph type: {name}")

    return graph, data, score


def get_random_graph(d, D, n, m, rng=default_rng(), latent_structure="random"):
    if latent_structure == "random":
        latent_nodes = ["h" + str(i) for i in range(d)]

    elif latent_structure == "random_chain_graph_c3":
        # To ensure that cliques are of size 3 and x_dim is equal to the number of cliques
        assert d % 3 == 0
        assert D == 2 * int(d / 3) - 1
        latent_nodes = ["h" + str(i) for i in range(d)]

    else:
        raise ValueError(f"Undefined latent structure: {latent_structure}")

    obs_nodes = ["x" + str(i) for i in range(D)]
    model = MarkovNetwork()
    model.add_nodes_from(latent_nodes + obs_nodes)

    if latent_structure == "random":
        # Random Graph
        edges = []
        is_edge_list = rng.binomial(1, 0.6, d**2)

        for e0_idx in range(d):
            for e1_idx in range(d):
                if is_edge_list[e0_idx * d + e1_idx] and e0_idx != e1_idx:
                    edges.append((latent_nodes[e0_idx], latent_nodes[e1_idx]))

    elif latent_structure == "G1":
        edges = [
            ("h0", "h1"),
            ("h0", "h2"),
            ("h1", "h2"),
            ("h2", "h3"),
            ("h3", "h4"),
            ("h3", "h5"),
            ("h4", "h5"),
        ]

    elif latent_structure == "random_chain_graph_c3":

        edges = []
        for e0_idx in range(d - 1):
            if e0_idx % 3 == 0:
                edges.append((latent_nodes[e0_idx], latent_nodes[e0_idx + 1]))
                edges.append((latent_nodes[e0_idx], latent_nodes[e0_idx + 2]))
            else:
                edges.append((latent_nodes[e0_idx], latent_nodes[e0_idx + 1]))

    else:
        raise ValueError(f"Undefined latent structure: {latent_structure}")

    # Adding edges between latent_nodes and obs_nodes
    for l in latent_nodes:
        edges += [(l, obs) for obs in obs_nodes]

    # Adding edges among obs_nodes themselevs
    for i in range(len(obs_nodes)):
        for j in range(i + 1, len(obs_nodes)):
            edges += [(obs_nodes[i], obs_nodes[j])]

    model.add_edges_from(edges)
    cliques = list(map(set, nx.find_cliques(model.triangulate())))
    # we cover all variables in a clique, i.e., both x and h
    cliques_indexes = [get_index_rep(clique, model)[0] for clique in cliques]
    cliques = [get_index_rep(clique, model)[1] for clique in cliques]

    if latent_structure == "random":
        potential_fns = [rng.random(2 ** (len(list(clique)))) for clique in cliques]
    elif latent_structure == "G1":
        potential_fns = [
            0.01 * np.ones((2 ** (len(list(clique))))) for clique in cliques
        ]
        potential_fns[0][0] = 2  # x0=0, x1=0, x2=0 -> h0=0, h1=0, h2=0
        potential_fns[1][-1] = 2  # x0=1, x1=1, x2=1 -> h2=1, h3=1
        potential_fns[2][2**D + 1] = 2  # x0=0, x1=0, x2=1 -> h3=0, h4=0, h5=1

    elif latent_structure == "random_chain_graph_c3":
        potential_fns = [
            0.01 * np.ones((2 ** (len(list(clique))))) for clique in cliques
        ]

        for i in range(D):
            # x_i = 1 iff clique c_i is activated (i.e. the values of its nodes are all 1)
            potential_fns[i][-(2**D) + 2 ** (D - 1 - i)] = 2

    else:
        raise ValueError(f"Undefined latent structure: {latent_structure}")

    factors_list = [
        DiscreteFactor(
            list(clique),
            [2] * len(list(clique)),
            potential_fns[clique_idx],
        )
        for clique_idx, clique in enumerate(cliques)
    ]

    model.add_factors(*factors_list)
    gibbs = GibbsSampling(model)
    train_data = gibbs.sample(size=n)
    eval_data = gibbs.sample(size=m)

    return model, (cliques_indexes, factors_list), (train_data, eval_data)


def get_potential_fns(model: MarkovNetwork, unobserved_cliques: list):
    """
    Given the markov model and a mutable representation of unfinished cliques,
    return a list of potential functions.

    Inputs
    --------
    model : MarkovNetwork

    unobserved_cliques : list
        A list of sets, where each set correspond to a clique. Each
        variable is represented by its index, an integer, Fully observed
        and cashed out variables are excluded from these sets.

    Outputs
    --------
    clique_potentials : list
        A list of potential functions for unobserved_cliques cliques
    """

    clique_potentials = []
    num_cliques = len(unobserved_cliques)
    nodes = [n for n in model.nodes]
    for c_ind in range(num_cliques):
        clique = list(unobserved_cliques[c_ind])
        clique_potentials.append(
            factor_sum_product(
                output_vars=[nodes[i] for i in clique], factors=model.factors
            )
        )

    return clique_potentials


def get_energy_fns(model: MarkovNetwork, full_cliques: list):
    """
    Given the markov model and a mutable representation of unfinished cliques,
    return a list of energy functions.

    Inputs
    --------
    model : MarkovNetwork

    full_cliques : list
        A list of sets, where each set correspond to a clique. Each
        variable is represented by its index, an integer.

    Outputs
    --------
    clique_energies : list
        A list of energy functions for full_cliques
    """
    clique_energies = []
    for c_ind in range(len(full_cliques)):
        clique_energies.append(-np.log(model.factors[c_ind].values))

    return clique_energies


def get_index_rep(nodes, model):
    all_nodes = [n for n in model.nodes]
    index = set([all_nodes.index(n) for n in nodes])
    nodes = [all_nodes[i] for i in list(index)]
    return index, nodes


def get_str_rep(nodes, model):
    all_nodes = [n for n in model.nodes]
    return set([sorted(all_nodes)[n] for n in nodes])


def find_incomplete_clique(
    observed, h_dim, necessary_has_incomplete_clique, clique_size=3
):
    incomplete_clique_ix = -1
    for i in range(h_dim // clique_size):
        n_observed_vars = np.sum(observed[clique_size * i : clique_size * (i + 1)])
        if n_observed_vars > 0 and n_observed_vars < clique_size:
            incomplete_clique_ix = i

    if necessary_has_incomplete_clique:
        assert (
            incomplete_clique_ix != -1
        )  # This function is only called when there is at least one incomplete clique
        # so we assert for debugging purposes

    incomplete_clique = {
        node
        for node in range(
            clique_size * incomplete_clique_ix,
            clique_size * (incomplete_clique_ix + 1),
        )
    }
    if incomplete_clique_ix == -1:
        incomplete_clique = {}
    return incomplete_clique


def get_chain_clique_selection_mask(
    gfn_state: tuple, K: int, h_dim: int, clique_size=3
):

    assert len(gfn_state) == 3
    assert len(gfn_state[0]) == len(gfn_state[1])
    assert len(gfn_state[0]) == len(gfn_state[2])
    assert len(np.unique(gfn_state[0])) <= 2
    assert len(np.unique(gfn_state[2])) <= 2
    assert np.max(gfn_state[1]) <= K

    N = len(gfn_state[0])
    x_dim = N - h_dim
    mask = np.zeros(N)

    observed_vars = np.nonzero(gfn_state[0])[0].flatten()

    num_latent_observed_vars = len(observed_vars) - x_dim

    if num_latent_observed_vars % clique_size == 0:
        mask = 1 - gfn_state[0]
    else:
        incomplete_clique = find_incomplete_clique(gfn_state[0], h_dim, True)

        eligible_vars = incomplete_clique - set(observed_vars)
        eligible_vars = list(filter(lambda x: x < h_dim, eligible_vars))
        mask[np.array(eligible_vars)] = 1

    return mask


def get_clique_selection_mask(
    gfn_state: tuple, unobserved_cliques: list, K: int, h_dim: int
):
    """
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
        never been cashed out as a part of a energy term.


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
    """
    assert len(gfn_state) == 3
    assert len(gfn_state[0]) == len(gfn_state[1])
    assert len(gfn_state[0]) == len(gfn_state[2])
    assert len(np.unique(gfn_state[0])) <= 2
    assert len(np.unique(gfn_state[2])) <= 2
    assert np.max(gfn_state[1]) <= K

    N = len(gfn_state[0])
    active_vars = set(np.nonzero(gfn_state[2] & gfn_state[0])[0].flatten())
    observed_vars = set(np.nonzero(gfn_state[0])[0].flatten())
    eligible_cliques = list(
        filter(lambda c: c.issuperset(active_vars), unobserved_cliques)
    )
    eligible_vars = (
        set().union(*eligible_cliques) - set(active_vars) - set(observed_vars)
    )

    mask = np.zeros(N)
    eligible_vars = list(filter(lambda x: x < h_dim, eligible_vars))
    if len(eligible_vars) == 0 or len(active_vars) == 0:
        mask = 1 - gfn_state[0]
    else:
        mask[np.array(list(eligible_vars))] = 1
    return mask


def get_value_policy_energy(
    gfn_state: tuple,
    unobserved_cliques: list,
    full_cliques: list,
    clique_potentials: list,
    K: int,
    count_partial_cliques: bool = False,
    graph: MarkovNetwork = None,
):
    """
    Given a GFN state, a mutable representation of unfinished cliques,
    a aligned list of clique potential functions, and the number of values
    return a new GFN state, an updated representation of unfinished cliques,
    and a scalar energy.

    Inputs
    --------
    gfn_state : tuple
        There are three iterables of the same length (N) in this tuple.
        The first iterable is binary and denotes observed variables.
        The second iteration can take on K+1 values and denote the
        if a value has been sampled for each observed variable, and
        if so, what that value is.
        The third iterable is binary and denotes if a variable has
        never been cashed out as a part of a energy term.

    unobserved_cliques : list
        A list of sets, where each set correspond to a clique. Each
        variable is represented by its index, an integer. Fully observed
        and cashed out variables are excluded from these sets.

    full_cliques : list
        A list of sets, where each set correspond to a clique. Each
        variable is represented by its index, an integer.

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
    energy : float
        The energy term associated with the clique that has been cashed out.
    """
    assert len(gfn_state) == 3
    assert len(gfn_state[0]) == len(gfn_state[1])
    assert len(gfn_state[0]) == len(gfn_state[2])
    assert len(np.unique(gfn_state[0])) <= 2
    assert len(np.unique(gfn_state[2])) <= 2
    assert len(unobserved_cliques) == len(clique_potentials)
    assert len(unobserved_cliques) == len(full_cliques)
    assert np.max(gfn_state[1]) <= K

    # we remove fully observed nodes
    all_observed_vars = set(np.nonzero(gfn_state[0])[0].flatten())
    new_unobserved_cliques = [c for c in unobserved_cliques]

    # we cash in every clique we complete and update the GFN state
    num_cliques = len(unobserved_cliques)
    energy = 0.0

    for c_ind in range(num_cliques):
        if len(unobserved_cliques[c_ind]) > 0:
            if all_observed_vars.issuperset(unobserved_cliques[c_ind]):
                new_unobserved_cliques[c_ind] = set()
                gfn_state[2][np.array(list(full_cliques[c_ind]))] = 0
                if isinstance(clique_potentials[c_ind], DiscreteFactor):
                    energy -= np.log(
                        clique_potentials[c_ind].values[
                            tuple(
                                gfn_state[1][
                                    np.array(sorted(list(full_cliques[c_ind])))
                                ]
                            )
                        ]
                    )
                else:
                    energy -= np.log(
                        clique_potentials[c_ind](
                            gfn_state[1][np.array(sorted(list(full_cliques[c_ind])))]
                        )
                    )
            elif count_partial_cliques:
                # we also cash out partially observed cliques
                assert graph is not None
                partially_observed_vars = list(
                    full_cliques[c_ind].intersection(all_observed_vars)
                )
                energy -= np.log(
                    factor_sum_product(
                        output_vars=list(get_str_rep(partially_observed_vars, graph)),
                        factors=[graph.factors[c_ind]],
                    ).values[
                        tuple(
                            gfn_state[1][
                                np.array(sorted(list(partially_observed_vars)))
                            ]
                        )
                    ]
                )

    if np.all(gfn_state[0] == 1):
        assert np.all([len(c) == 0 for c in new_unobserved_cliques])
    return gfn_state, new_unobserved_cliques, energy


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
    gfn_state = (
        np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 2, 2, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]

    mask = get_clique_selection_mask(gfn_state, unobserved_cliques, K)
    assert np.all(mask == np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])), mask

    # Setting up a list of dummy potential functions
    clique_potentials = [lambda c: 0.2 if len(c) == 7 else np.nan, lambda c: -2]
    gfn_state = (
        np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]
    full_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, unobserved_cliques, full_cliques, clique_potentials, K
    )
    assert energy == -np.log(0.2)
    assert len(new_unobserved_cliques[0]) == 0
    gfn_state = (
        np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 0, 2, 0, 1, 0, 1]),
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
    )
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, new_unobserved_cliques, full_cliques, clique_potentials, K
    )
    assert energy == 0.0

    # Test potential fns
    model, (full_cliques, _), data = get_random_graph(d=6, D=4, n=4)
    unobserved_cliques = full_cliques.copy()
    gfn_state = (
        np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )

    clique_potentials = get_potential_fns(model, unobserved_cliques)
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, unobserved_cliques, full_cliques, clique_potentials, K
    )

    target_energy = 0
    for c_ind, c in enumerate(new_unobserved_cliques):
        if len(c) == 0:
            target_energy += clique_potentials[c_ind].values[
                (tuple(gfn_state[1][np.array(sorted(list(full_cliques[c_ind])))]))
            ]
    assert energy == target_energy

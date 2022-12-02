import numpy as np
import jax
import jax.numpy as jnp
import jraph
from collections import namedtuple
from itertools import permutations

Graph = namedtuple("Graph", ["structure", "values"])


def to_graphs_tuple(
    full_cliques: list, gfn_state: tuple, K: int, pad: bool = True
) -> Graph:
    """Converts a tuple representation of the GFN state into a `Graph` object
    compatible with the input type of the clique and value policies.

    We use the following conversion table to decide the .node attribute
    from each output (structure_graph or value_graph) GraphsTuple:

        0, 1, ..., N-1 -> identifies the observed nodes in the structure_graph
            (corresponds to their original indices).

        N, N+1, ..., N+K-1 -> identifies the values of the observed nodes
            in the value_graph, where N+i means value i.

        N+K -> identifies unobserved nodes in both the structure_graph
        and in the value_graph. These are the dummy indices.

        N+K+1 -> identifies the *unique* node whose value needs to be
        sampled by the value policy. This is only part of the value_graph and
        corresponds to the last node added by the clique policy.


    Parameters
    ----------
    full_cliques: list
        A list of sets, where each set correspond to a clique. Each
        variable is represented by its index, an integer.
    gfn_state : tuple
        There are three iterables of the same length (N) in this tuple.
        The first iterable is binary and denotes observed variables.
        The second iteration can take on K+1 values and denote the
        if a value has been sampled for each observed variable, and
        if so, what that value is.
        The third iterable is binary and denotes if a variable has
        never been cashed out as a part of a reward term.
    K : int
        The number of possible values.
    pad : bool, optional
        Whether to pad the resulting graphs in `Graph` with zeroes
        to guarantee a fixed size and prevent re-compilation of the
        clique and value policies, by default True.

    Returns
    -------
    Graph
        Representation of the state containing the two GraphsTuples:
        one for the values and one for the structure.
    """
    squeezed_states = []    
    for i in range(len(gfn_state)):
        squeezed_states.append(np.squeeze(gfn_state[i]))
        assert len(squeezed_states[i].shape) == 1

    gfn_state = (squeezed_states[0], squeezed_states[1], squeezed_states[2])
    
    num_variables = gfn_state[0].shape[0]
    structure_node_features = np.arange(num_variables)
    structure_node_features = np.where(
        gfn_state[0] == 0, num_variables + K, structure_node_features
    )

    edges = []
    for clique in full_cliques:
        clique_edges = permutations(clique, r=2)
        edges.extend(clique_edges)

    """
    Filtering out duplicate edges, which can happen if two cliques have edges in common. 
    Then sorting in ascending order of senders.
    """
    edges = list(set(edges))
    edges.sort(key=lambda x: x[0])
    senders, receivers = zip(*edges)

    edge_features = np.ones_like(senders)

    structure_graph = jraph.GraphsTuple(
        nodes=structure_node_features,
        edges=edge_features,
        senders=np.array(senders),
        receivers=np.array(receivers),
        globals=None,
        n_node=np.array([num_variables]),
        n_edge=np.array([len(edges)]),
    )

    """
    Values need to have distinct embeddings than positions. Hence we shift 
    everything by num_variables
    """
    value_node_features = np.array(gfn_state[1]) + num_variables
    value_graph = jraph.GraphsTuple(
        nodes=value_node_features,
        edges=edge_features,
        senders=np.array(senders),
        receivers=np.array(receivers),
        globals=None,
        n_node=np.array([num_variables]),
        n_edge=np.array([len(edges)]),
    )
    if pad:
        # Necessary to avoid changing shapes too often, which triggers jax re-compilation
        structure_graph = pad_graph_to_nearest_power_of_two(structure_graph)
        value_graph = pad_graph_to_nearest_power_of_two(value_graph)

        structure_graph.nodes[num_variables:] = (
            num_variables + K
        )  # Index signaling dummy embedding
        value_graph.nodes[num_variables:] = num_variables + K

    return Graph(structure=structure_graph, values=value_graph)


def _nearest_bigger_power_of_two(x):
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(graphs_tuple):
    # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_edge))

    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )

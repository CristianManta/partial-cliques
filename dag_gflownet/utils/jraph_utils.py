import numpy as np
import jax
import jax.numpy as jnp
import jraph
from collections import namedtuple
from itertools import permutations

Graph = namedtuple("Graph", ["structure", "values"])


def to_graphs_tuple(
    full_cliques: list, gfn_states: list, K: int, x_dim: int, pad: bool = False
) -> Graph:
    """Converts a list of tuple representations of the GFN state into a `Graph` object
    compatible with the input type of the clique and value policies.

    We use the following conversion table to decide the .node attribute
    from each output (structure_graph or value_graph) GraphsTuple:

        0, 1, ..., N-1 -> identifies the nodes indices in the structure_graph
            (corresponds to their original indices).

        N, N+1, ..., N+K-1 -> identifies the values of the observed nodes
            in the value_graph, where N+i means value i.

        N+K -> identifies unobserved nodes in
        the value_graph. These are the dummy indices.

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
        never been cashed out as a part of a energy term.
    K : int
        The number of possible values.
    x_dim: int
        The number of x variables.
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
    structure_graphs = []
    value_graphs = []
    for gfn_state in gfn_states:
        squeezed_states = []
        for i in range(len(gfn_state)):
            squeezed_states.append(np.squeeze(gfn_state[i]))
            assert len(squeezed_states[i].shape) == 1

        gfn_state = (squeezed_states[0], squeezed_states[1], squeezed_states[2])

        num_variables = gfn_state[0].shape[0]
        h_dim = num_variables - x_dim
        structure_node_features = np.arange(num_variables)
        # structure_node_features = np.where(
        #    gfn_state[0] == 0, num_variables + K, structure_node_features
        # )

        edges = []

        # Adding all the edges in the cliques
        for i, clique in enumerate(full_cliques):
            clique = clique.union(
                set(range(h_dim, num_variables))
            )  # Extending the cliques to contain x
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

        structure_graphs.append(structure_graph)
        value_graphs.append(value_graph)

    structure_graphs = jraph.batch(structure_graphs)
    value_graphs = jraph.batch(value_graphs)

    if (
        pad
    ):  # TODO: I think that we don't need this in our setting anymore, since the edges don't depend on the gfn_state,
        # so the size of the GraphsTuple attributes should be constant

        # Necessary to avoid changing shapes too often, which triggers jax re-compilation
        structure_graphs = pad_graph_to_nearest_power_of_two(structure_graphs)
        value_graphs = pad_graph_to_nearest_power_of_two(value_graphs)

        batch_size = len(gfn_states)
        structure_graphs.nodes[batch_size * num_variables :] = (
            num_variables + K
        )  # Index signaling dummy embedding
        value_graphs.nodes[batch_size * num_variables :] = num_variables + K

        # This is to convert all numpy arrays to jnp.DeviceArray. It's a quirk of the padding which
        # re-converts jnp arrays back to numpy ones. In the future, I think that I'll get rid of padding entirely
        structure_graphs = jraph.batch([structure_graphs])
        value_graphs = jraph.batch([value_graphs])

    return Graph(structure=structure_graphs, values=value_graphs)


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

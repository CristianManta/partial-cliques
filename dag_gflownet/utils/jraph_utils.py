import numpy as np
import jax.numpy as jnp
import jraph


def to_graphs_tuple(adjacencies, pad=True):
    num_graphs, num_variables = adjacencies.shape[:2]
    n_node = np.full((num_graphs,), num_variables, dtype=np.int_)

    counts, senders, receivers = np.nonzero(adjacencies)
    n_edge = np.zeros((num_graphs,), dtype=np.int_)
    np.add.at(n_edge, counts, 1)

    # Node features: node indices
    # Edge features: binary features "is the edge in the original DAG?"
    nodes = np.tile(np.arange(num_variables), num_graphs)
    edges = np.ones_like(senders)

    graphs_tuple =  jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=n_node,
        n_edge=n_edge,
    )
    if pad:
        graphs_tuple = pad_graph_to_nearest_power_of_two(graphs_tuple)
    return graphs_tuple

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
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


def get_node_offsets(graphs):
    n_node = jnp.zeros_like(graphs.n_node)
    n_node = n_node.at[1:].set(graphs.n_node[:-1])
    offsets = jnp.cumsum(n_node)
    return jnp.repeat(offsets, graphs.n_edge, axis=0,
        total_repeat_length=graphs.edges.shape[0])

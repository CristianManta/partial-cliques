import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
import jraph
import haiku as hk

from dag_gflownet.utils.jraph_utils import Graph
from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy


@pytest.fixture
def setup():

    # Defining a batch of 2 structure graphs
    node_features1 = np.array([0, 1, 9, 9, 9, 9])
    senders1 = np.array([0, 1])
    receivers1 = np.array([1, 0])
    edges1 = np.array([1, 1])
    n_node1 = np.array([6])
    n_edge1 = np.array([2])
    structure_graph1 = jraph.GraphsTuple(
        nodes=node_features1,
        senders=senders1,
        receivers=receivers1,
        edges=edges1,
        n_node=n_node1,
        n_edge=n_edge1,
        globals=None,
    )

    node_features2 = np.array([9, 9, 9, 3, 4, 5])
    senders2 = np.array([3, 4, 4, 5, 5, 3])
    receivers2 = np.array([4, 3, 5, 4, 3, 5])
    edges2 = np.array([1, 1, 1, 1, 1, 1])
    n_node2 = np.array([6])
    n_edge2 = np.array([6])
    structure_graph2 = jraph.GraphsTuple(
        nodes=node_features2,
        senders=senders2,
        receivers=receivers2,
        edges=edges2,
        n_node=n_node2,
        n_edge=n_edge2,
        globals=None,
    )

    batched_structure_graphs = jraph.batch([structure_graph1, structure_graph2])

    # Defining a batch of 2 value graphs
    node_features1_val = np.array([6, 8, 9, 9, 9, 9])
    senders1_val = np.array([0, 1])
    receivers1_val = np.array([1, 0])
    edges1_val = np.array([1, 1])
    n_node1_val = np.array([6])
    n_edge1_val = np.array([2])
    value_graph1 = jraph.GraphsTuple(
        nodes=node_features1_val,
        senders=senders1_val,
        receivers=receivers1_val,
        edges=edges1_val,
        n_node=n_node1_val,
        n_edge=n_edge1_val,
        globals=None,
    )

    node_features2_val = np.array([9, 9, 9, 6, 7, 8])
    senders2_val = np.array([3, 4, 4, 5, 5, 3])
    receivers2_val = np.array([4, 3, 5, 4, 3, 5])
    edges2_val = np.array([1, 1, 1, 1, 1, 1])
    n_node2_val = np.array([6])
    n_edge2_val = np.array([6])
    value_graph2 = jraph.GraphsTuple(
        nodes=node_features2_val,
        senders=senders2_val,
        receivers=receivers2_val,
        edges=edges2_val,
        n_node=n_node2_val,
        n_edge=n_edge2_val,
        globals=None,
    )

    batched_value_graphs = jraph.batch([value_graph1, value_graph2])

    batch_size, max_nodes = 2, 6

    # Concatenating structure and values into the same data structure
    graphs = Graph(structure=batched_structure_graphs, values=batched_value_graphs)
    masks = jnp.ones((batch_size, max_nodes), dtype=int)
    return graphs, masks


def test_clique_policy_shapes_jit(setup):
    graphs, masks = setup
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(clique_policy))
    params = model.init(key, graphs, masks)

    # Applying the model
    log_policy_cliques = jax.jit(model.apply)(params, graphs, masks)

    assert log_policy_cliques.shape == (masks.shape[0], masks.shape[1] + 1)


def test_value_policy_shapes_jit(setup):
    graphs, masks = setup
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(value_policy))
    params = model.init(key, graphs, masks)

    # Applying the model
    log_policy_values, log_flows = jax.jit(model.apply)(params, graphs, masks)

    assert log_policy_values.shape == (masks.shape[0],)
    assert log_flows.shape == (masks.shape[0],)

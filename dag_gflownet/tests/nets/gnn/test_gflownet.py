import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
import jraph
import haiku as hk

from dag_gflownet.utils.jraph_utils import Graph, to_graphs_tuple
from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy, value_policy_MLP
from dag_gflownet.utils.data import get_clique_selection_mask


@pytest.fixture
def setup():
    K = 2
    batch_size = 1
    h_dim = 6
    x_dim = 4
    # Setting up a dummy GFN state
    # Assuming that we have 10 variables, x_0^3 and h_0^5
    # We have two cliques {x_0^3, h_0^2} and {x_0^3, h_3^5}
    # We have only fully observed x_0^3 and h_0
    gfn_state = (
        np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 2, 2, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    full_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]

    graphs = to_graphs_tuple(full_cliques, [gfn_state], K, x_dim, pad=False)
    mask = jnp.expand_dims(
        jnp.array(get_clique_selection_mask(gfn_state, unobserved_cliques, K, h_dim)), 0
    )
    return graphs, mask, x_dim, K


@pytest.fixture
def setup_MLP():
    K = 2
    batch_size = 16
    h_dim = 1
    x_dim = 1
    # Setting up a dummy GFN state
    # Assuming that we have 10 variables, x_0^3 and h_0^5
    # We have two cliques {x_0^3, h_0^2} and {x_0^3, h_3^5}
    # We have only fully observed x_0^3 and h_0
    gfn_state = (
        np.array([0, 1]),
        np.array([2, 0]),
        np.array([1, 1]),
    )
    full_cliques = [set([0, 1])]
    unobserved_cliques = [set([0, 1])]

    graphs = to_graphs_tuple(
        full_cliques, [gfn_state] * batch_size, K, x_dim, pad=False
    )
    mask = jnp.ones((batch_size, x_dim + h_dim))  # Only the shape matters this time
    return graphs, mask, x_dim, K


def test_clique_policy_shapes_jit(setup):
    graphs, masks, x_dim, K = setup
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(clique_policy))
    sampling_method = 1
    params = model.init(key, graphs, masks, x_dim, K, sampling_method)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4, 5))
    log_policy_cliques = forward(params, graphs, masks, x_dim, K, sampling_method)

    assert log_policy_cliques.shape == (masks.shape[0], masks.shape[1] - x_dim + 1)

    sampling_method = 2
    params = model.init(key, graphs, masks, x_dim, K, sampling_method)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4, 5))
    log_policy_cliques = forward(params, graphs, masks, x_dim, K, sampling_method)

    assert log_policy_cliques.shape == (masks.shape[0], masks.shape[1] - x_dim + 1)

    sampling_method = 3
    params = model.init(key, graphs, masks, x_dim, K, sampling_method)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4, 5))
    log_policy_cliques = forward(params, graphs, masks, x_dim, K, sampling_method)

    assert log_policy_cliques.shape == (masks.shape[0], masks.shape[1] - x_dim + 1)


def test_value_policy_shapes_jit(setup):
    graphs, masks, x_dim, K = setup
    val_graph = graphs.values._replace(
        nodes=graphs.values.nodes.at[1].set(
            masks.shape[1] + K + 1
        )  # Let's say that we want to sample the value for node at index 1
    )
    structure_graph = graphs.structure._replace(
        nodes=graphs.structure.nodes.at[1].set(1)
    )
    graphs = Graph(structure=structure_graph, values=val_graph)
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(value_policy))
    params = model.init(key, graphs, masks, x_dim, K)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4))
    log_policy_values, log_flows = forward(params, graphs, masks, x_dim, K)

    assert log_policy_values.shape == (masks.shape[0], K)
    assert log_flows.shape == (masks.shape[0],)


def test_value_policy_MLP(setup_MLP):
    graphs, masks, x_dim, K = setup_MLP
    val_graph = graphs.values._replace(
        nodes=graphs.values.nodes.at[1].set(
            masks.shape[1] + K + 1
        )  # Let's say that we want to sample the value for node at index 1
    )
    structure_graph = graphs.structure._replace(
        nodes=graphs.structure.nodes.at[1].set(
            1
        )  # TODO: Throughout building the models, I assumed that the node would still be classified as unobserved
    )  # until it gets its concrete value. It seems that I haven't been consistent
    graphs = Graph(structure=structure_graph, values=val_graph)
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(value_policy_MLP))
    params = model.init(key, graphs, masks, x_dim, K)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4))
    log_policy_values, log_flows = forward(params, graphs, masks, x_dim, K)

    assert log_policy_values.shape == (masks.shape[0], K)
    assert log_flows.shape == (masks.shape[0],)

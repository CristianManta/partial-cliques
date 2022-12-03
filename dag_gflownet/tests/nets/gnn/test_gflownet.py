import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
import jraph
import haiku as hk

from dag_gflownet.utils.jraph_utils import Graph, to_graphs_tuple
from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy
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

    graphs = to_graphs_tuple(full_cliques, gfn_state, K, pad=True)
    mask = jnp.expand_dims(
        jnp.array(get_clique_selection_mask(gfn_state, unobserved_cliques, K)), 0
    )
    return graphs, mask, x_dim, K


def test_clique_policy_shapes_jit(setup):
    graphs, masks, x_dim, K = setup
    seed = 0
    key = random.PRNGKey(seed)

    # Initializing the model
    model = hk.without_apply_rng(hk.transform(clique_policy))
    params = model.init(key, graphs, masks, x_dim, K)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(3, 4))
    log_policy_cliques = forward(params, graphs, masks, x_dim, K)

    assert log_policy_cliques.shape == (masks.shape[0], masks.shape[1] - x_dim + 1)


def test_value_policy_shapes_jit(setup):
    graphs, masks, x_dim, K = setup
    graphs.values.nodes[1] = masks.shape[1] + K + 1 # Let's say that we want to sample the value for node at index 1
    graphs.structure.nodes[1] = 1
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

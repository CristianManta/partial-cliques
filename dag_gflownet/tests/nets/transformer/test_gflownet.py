import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
import jraph
import haiku as hk

from dag_gflownet.utils.jraph_utils import Graph, to_graphs_tuple
from dag_gflownet.nets.transformer.gflownet import value_policy_transformer


@pytest.fixture
def setup():
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


def test_value_policy_transformer(setup):
    graphs, masks, x_dim, K = setup
    batch_size = masks.shape[0]
    num_variables = masks.shape[1]
    for i in range(batch_size):
        new_nodes = graphs.values.nodes.at[i * num_variables].set(num_variables + K + 1)

        graphs = Graph(
            structure=graphs.structure,
            values=graphs.values._replace(nodes=new_nodes),
        )

    seed = 0
    key = random.PRNGKey(seed)
    next_key, _ = jax.random.split(key, 2)

    # Initializing the model
    model = hk.transform(value_policy_transformer)
    params = model.init(key, graphs, masks, x_dim, K)

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(4, 5))
    log_policy_values, log_flows = forward(params, next_key, graphs, masks, x_dim, K)

    assert log_policy_values.shape == (masks.shape[0], K)
    assert log_flows.shape == (masks.shape[0],)

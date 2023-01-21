import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
import jraph
import haiku as hk

from dag_gflownet.utils.jraph_utils import Graph, to_graphs_tuple
from dag_gflownet.nets.transformer.gflownet import (
    value_policy_transformer,
    clique_policy_transformer,
)


@pytest.fixture
def setup():
    K = 2
    batch_size = 1
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


def test_clique_policy_transformer(setup):
    graphs, masks, x_dim, K = setup
    batch_size = masks.shape[0]
    num_variables = masks.shape[1]
    for i in range(batch_size):
        if i == 1:  # Suppose that we don't observe the second "h" in the batch
            continue
        new_nodes = graphs.values.nodes.at[i * num_variables].set(num_variables + K + 1)

        graphs = Graph(
            structure=graphs.structure,
            values=graphs.values._replace(nodes=new_nodes),
        )

    seed = 0
    key = random.PRNGKey(seed)
    next_key, _ = jax.random.split(key, 2)

    # Initializing the model
    model = hk.transform(clique_policy_transformer)
    params = model.init(
        key,
        np.array([[0, 1]]),
        np.repeat(np.array(graphs.values.nodes).reshape(1, -1), batch_size, axis=0),
        masks,
        x_dim,
        K,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        key_size=32,
        dropout_rate=0.0,
    )

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(5, 6, 7, 8, 9, 10, 11))
    logits = forward(
        params,
        next_key,
        np.array([[0, 1]]),
        np.repeat(np.array(graphs.values.nodes).reshape(1, -1), batch_size, axis=0),
        masks,
        x_dim,
        K,
        128,
        4,
        6,
        32,
        0.0,
    )
    h_dim = num_variables - x_dim

    assert logits.shape == (masks.shape[0], h_dim)


def test_value_policy_transformer(setup):
    graphs, masks, x_dim, K = setup
    batch_size = masks.shape[0]
    num_variables = masks.shape[1]
    for i in range(batch_size):
        if i == 1:  # Suppose that we don't observe the second "h" in the batch
            continue
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
    params = model.init(
        key,
        np.repeat(np.array(graphs.values.nodes).reshape(1, -1), batch_size, axis=0),
        masks,
        x_dim,
        K,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        key_size=32,
        dropout_rate=0.0,
    )

    # Applying the model
    forward = jax.jit(model.apply, static_argnums=(4, 5, 6, 7, 8, 9, 10))
    log_policy_values, log_flows = forward(
        params,
        next_key,
        np.repeat(np.array(graphs.values.nodes).reshape(1, -1), batch_size, axis=0),
        masks,
        x_dim,
        K,
        128,
        4,
        6,
        32,
        0.0,
    )

    assert log_policy_values.shape == (masks.shape[0], K)
    assert log_flows.shape == (masks.shape[0],)

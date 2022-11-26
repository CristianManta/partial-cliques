import pytest
import numpy as np
from jax import jit
import jax.numpy as jnp

from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


# def test_to_graphs_tuple_jit():
#     K = 2
#     # Setting up a dummy GFN state
#     # Assuming that we have 10 variables, x_0^3 and h_0^5
#     # We have two cliques {x_0^3, h_0^2} and {x_0^3, h_3^5}
#     # We have only fully observed x_0^3 and h_0
#     gfn_state = (
#         jnp.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
#         jnp.array([1, 2, 2, 2, 2, 2, 0, 1, 0, 1]),
#         jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
#     )

#     result = jit(to_graphs_tuple)(gfn_state, K, pad=True)
#     pass


def test_to_graphs_tuple_concrete():
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
    full_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]

    result = to_graphs_tuple(full_cliques, gfn_state, K, pad=True)

    np.testing.assert_array_equal(result.structure.n_node, np.array([10, 7]))
    np.testing.assert_array_equal(result.values.n_node, np.array([10, 7]))
    np.testing.assert_array_equal(result.structure.n_edge, np.array([72, 56]))
    np.testing.assert_array_equal(result.values.n_edge, np.array([72, 56]))
    assert len(result.structure.nodes) == 17
    assert len(result.values.nodes) == 17
    assert len(result.structure.edges) == 128
    assert len(result.values.edges) == 128

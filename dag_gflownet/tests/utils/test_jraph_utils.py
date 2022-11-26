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

    """
    The original graph above has 10 nodes and 36 undirected edges. We convert this 
    into two separate GraphsTuple objects (result.structure and result.values) 
    because the gfns need to take the sum of positional and value embeddings 
    (which requires two containers for the nodes features, only possible with 
    two GraphsTuple objects). 
    
    Each of these two graphs have been padded by another graph 
    so that the total number of nodes is the closest higher power of two + 1 
    and the number of edges is the closest higher power of two. 
    This is because the two components in the padded graph are part of a 
    single data structure (feel free to inspect the values), and we want 
    the shape of it to not change too often, otherwise JAX will re-compile 
    and this will slow things down. This is why n_node must be [10, 7], for 
    a total of 17 = 2^4 + 1 nodes. 
    
    Regarding the edges, each undirected edge implies two directed ones, thus 
    there are 72 *directed* edges in our original graph. With the padding, we 
    have 128 = 72 + 56 edges.    
    """
    np.testing.assert_array_equal(result.structure.n_node, np.array([10, 7]))
    np.testing.assert_array_equal(result.values.n_node, np.array([10, 7]))
    np.testing.assert_array_equal(result.structure.n_edge, np.array([72, 56]))
    np.testing.assert_array_equal(result.values.n_edge, np.array([72, 56]))
    assert len(result.structure.nodes) == 17
    assert len(result.values.nodes) == 17
    assert len(result.structure.edges) == 128
    assert len(result.values.edges) == 128

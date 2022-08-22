import pytest
import numpy as np

from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


class TestToGraphsTuple:
    @pytest.fixture
    def graphs(self):
        adjacencies = np.array([
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [1, 0, 0]],

            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        ])
        return to_graphs_tuple(adjacencies, pad=False)

    def test_type(self, graphs):
        assert isinstance(graphs, GraphsTuple)

    def test_n_node(self, graphs):
        assert graphs.n_node.shape == (4,)
        expected_n_node = np.array([3, 3, 3, 3], dtype=np.int_)
        np.testing.assert_array_equal(graphs.n_node, expected_n_node)

    def test_n_edge(self, graphs):
        assert graphs.n_edge.shape == (4,)
        expected_n_edge = np.array([1, 0, 2, 3], dtype=np.int_)
        np.testing.assert_array_equal(graphs.n_edge, expected_n_edge)

    def test_nodes(self, graphs):
        assert graphs.nodes.shape == (12,)
        expected_nodes = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int_)
        np.testing.assert_array_equal(graphs.nodes, expected_nodes)

    def test_edges(self, graphs):
        assert graphs.edges.shape == (6,)
        expected_edges = np.ones((6,), dtype=np.int_)
        np.testing.assert_array_equal(graphs.edges, expected_edges)

    def test_senders(self, graphs):
        assert graphs.senders.shape == (6,)
        expected_senders = np.array([1, 7, 8, 9, 10, 11], dtype=np.int_)
        np.testing.assert_array_equal(graphs.senders, expected_senders)

    def test_receivers(self, graphs):
        assert graphs.receivers.shape == (6,)
        expected_receivers = np.array([2, 6, 6, 10, 11, 9], dtype=np.int_)
        np.testing.assert_array_equal(graphs.receivers, expected_receivers)

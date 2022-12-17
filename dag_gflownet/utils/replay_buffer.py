import numpy as np
import math

from numpy.random import default_rng
from collections import namedtuple
from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple

Graph = namedtuple("Graph", ["structure", "values"])


class ReplayBuffer:
    # TODO: Change this class depending on whether we want to store whole transitions in the replay buffer
    def __init__(self, capacity, full_cliques, K, num_variables, x_dim):
        self.capacity = capacity
        self.num_variables = num_variables
        self.full_cliques = full_cliques
        self.K = K
        self.x_dim = x_dim

        dtype = np.dtype(
            [
                ("observed", np.bool, (num_variables,)),
                ("values", np.int, (num_variables,)),
                ("cashed", np.bool, (num_variables,)),
                ("actions", np.int_, (2,)),
                ("is_exploration", np.bool_, (1,)),
                ("done", np.bool_, (1,)),
                ("value_energies", np.float_, (1,)),
                ("var_energies", np.float_, (1,)),
                ("mask", np.bool, (num_variables,)),
                ("next_mask", np.bool, (num_variables,)),
                ("next_observed", np.bool, (num_variables,)),
                ("next_values", np.int, (num_variables,)),
                ("next_cashed", np.bool, (num_variables,)),
            ]
        )
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)

    def add(
        self, observations, actions, is_exploration, next_observations, energies, dones
    ):

        (var_energies, value_energies) = energies

        bsz = len(observations["gfn_state"])

        for i in range(bsz):
            # num_samples = np.sum(~dones)
            add_idx = self._index
            self._index = (self._index + 1) % self.capacity
            self._is_full |= self._index == self.capacity - 1
            # self._index = (self._index + num_samples) % self.capacity
            # indices[~dones] = add_idx

            data = {
                "observed": observations["gfn_state"][i][0],
                "values": observations["gfn_state"][i][1],
                "cashed": observations["gfn_state"][i][2],
                "done": np.array([next_observations["is_done"][i]]),
                "next_observed": next_observations["gfn_state"][i][0],
                "next_values": next_observations["gfn_state"][i][1],
                "next_cashed": next_observations["gfn_state"][i][2],
                "actions": actions[i],
                "var_energies": np.array([var_energies[i]]),
                "value_energies": np.array([value_energies[i]]),
                "mask": observations["mask"][i],
                "next_mask": next_observations["mask"][i]
                # Extra keys for monitoring
            }

            for name in data:
                shape = self._replay.dtype[name].shape
                self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))

    def sample(self, batch_size, rng=default_rng()):
        # TODO
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        observed = [sample["observed"] for sample in samples]
        values = [sample["values"] for sample in samples]
        mask = [sample["mask"] for sample in samples]
        next_mask = [sample["next_mask"] for sample in samples]
        gfn_state = [
            (sample["observed"], sample["values"], sample["cashed"])
            for sample in samples
        ]

        next_observed = [sample["next_observed"] for sample in samples]
        next_values = [sample["next_values"] for sample in samples]
        actions = [sample["actions"] for sample in samples]
        dones = [sample["done"] for sample in samples]
        var_energies = [sample["var_energies"] for sample in samples]
        value_energies = [sample["value_energies"] for sample in samples]
        next_gfn_state = [
            (sample["next_observed"], sample["next_values"], sample["next_cashed"])
            for sample in samples
        ]

        graphs_tuple = to_graphs_tuple(self.full_cliques, gfn_state, self.K, self.x_dim)
        next_graphs_tuple = to_graphs_tuple(
            self.full_cliques, next_gfn_state, self.K, self.x_dim
        )

        # Flagging the selected node for each action of the clique policy
        for i in range(batch_size):
            if actions[i][0] != -1:
                """
                new_nodes = graphs_tuple.values.nodes.at[
                    i * self.num_variables + actions[i][0]
                ].set(self.num_variables + self.K + 1)
                graphs_tuple = Graph(
                    structure=graphs_tuple.structure,
                    values=graphs_tuple.values._replace(nodes=new_nodes),
                )
                """
                values[i][actions[i][0]] = self.K + 1

        # Convert structured array into dictionary
        # If we find that the training loop is too slow, we might want to
        # store the graphs tuples using replay.add directly by storing each
        # of its attributes separately (ugly solution, but saves performance)
        return {
            "observed": np.stack(observed, axis=0),
            "values": np.stack(values, axis=0),
            "next_observed": np.stack(next_observed, axis=0),
            "next_values": np.stack(next_values, axis=0),
            "graphs_tuple": graphs_tuple,
            "next_graphs_tuple": next_graphs_tuple,
            "actions": np.stack(actions, axis=0),
            "dones": np.stack(dones, axis=0),
            "var_energies": np.stack(var_energies, axis=0),
            "value_energies": np.stack(value_energies, axis=0),
            "mask": np.stack(mask, axis=0),
            "next_mask": np.stack(next_mask),
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[: len(self)]

    def save(self, filename):
        data = {
            "version": 3,
            "replay": self.transitions,
            "index": self._index,
            "is_full": self._is_full,
            "prev": self._prev,
            "capacity": self.capacity,
            "num_variables": self.num_variables,
        }
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            data = np.load(f)
            if data["version"] != 3:
                raise IOError(f'Unknown version: {data["version"]}')
            replay = cls(capacity=data["capacity"], num_variables=data["num_variables"])
            replay._index = data["index"]
            replay._is_full = data["is_full"]
            replay._prev = data["prev"]
            replay._replay[: len(replay)] = data["replay"]
        return replay

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables**2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables**2)
        decoded = decoded.reshape(
            *encoded.shape[:-1], self.num_variables, self.num_variables
        )
        return decoded.astype(dtype)

    @property
    # TODO: do this properly
    def dummy(self):
        shape = (1, self.num_variables, self.num_variables)
        structure_graph = GraphsTuple(
            nodes=np.arange(self.num_variables),
            edges=np.zeros((1,), dtype=np.int_),
            senders=np.zeros((1,), dtype=np.int_),
            receivers=np.zeros((1,), dtype=np.int_),
            globals=None,
            n_node=np.full((1,), self.num_variables, dtype=np.int_),
            n_edge=np.ones((1,), dtype=np.int_),
        )

        value_graph = GraphsTuple(
            nodes=np.arange(self.num_variables),
            edges=np.zeros((1,), dtype=np.int_),
            senders=np.zeros((1,), dtype=np.int_),
            receivers=np.zeros((1,), dtype=np.int_),
            globals=None,
            n_node=np.full((1,), self.num_variables, dtype=np.int_),
            n_edge=np.ones((1,), dtype=np.int_),
        )

        return {
            "graph": Graph(structure=structure_graph, values=value_graph),
            "values": np.zeros((1, self.num_variables), dtype=np.int),
            "value_energy": np.zeros((1, 1), dtype=np.float_),
            "clique_energy": np.zeros((1, 1), dtype=np.float_),
            "mask": np.zeros(
                (
                    1,
                    self.num_variables,
                ),
                dtype=np.bool,
            ),
            "next_mask": np.zeros((1, self.num_variables), dtype=np.bool),
        }

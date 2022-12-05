import numpy as np
import math

from numpy.random import default_rng
from collections import namedtuple
from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple

Graph = namedtuple("Graph", ["structure", "values"])


class ReplayBuffer:
    # TODO: Change this class depending on whether we want to store whole transitions in the replay buffer
    def __init__(self, capacity, full_cliques, K, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables
        self.full_cliques = full_cliques
        self.K = K

        dtype = np.dtype(
            [
                ("observed", np.bool, (num_variables,)),
                ("values", np.int, (num_variables,)),
                ("cashed", np.bool, (num_variables,)),
                ("actions", np.int_, (2,)),
                ("is_exploration", np.bool_, (1,)),
                ("done", np.bool_, (1,)),
                ("value_rewards", np.float_, (1,)),
                ("var_rewards", np.float_, (1,)),
                ("mask", np.uint8, (num_variables,)),
                ("next_mask", np.uint8, (num_variables,)),
                ("next_observed", np.bool, (num_variables,)),
                ("next_values", np.bool, (num_variables,)),
                ("next_cashed", np.bool, (num_variables,)),
            ]
        )
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)

    def add(
        self, observations, actions, is_exploration, next_observations, rewards, dones
    ):

        (var_rewards, value_rewards) = rewards

        # num_samples = np.sum(~dones)
        add_idx = self._index
        self._index = (self._index + 1) % self.capacity
        self._is_full |= self._index == self.capacity - 1
        # self._index = (self._index + num_samples) % self.capacity
        # indices[~dones] = add_idx

        data = {
            "observed": observations["gfn_state"][0],
            "values": observations["gfn_state"][1],
            "cashed": observations["gfn_state"][2],
            "done": np.array([next_observations["is_done"]]),
            "next_observed": next_observations["gfn_state"][0],
            "next_values": next_observations["gfn_state"][1],
            "next_cashed": next_observations["gfn_state"][2],
            "actions": actions,
            "var_rewards": np.array([var_rewards]),
            "value_rewards": np.array([value_rewards]),
            "mask": observations["mask"],
            "next_mask": next_observations["mask"]
            # Extra keys for monitoring
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))

    def sample(self, batch_size, rng=default_rng()):
        # TODO
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        observed = samples["observed"]
        values = samples["values"]
        cashed = samples["cashed"]
        gfn_state = (observed, values, cashed)

        next_observed = samples["next_observed"]
        next_values = samples["next_values"]
        next_cashed = samples["next_cashed"]
        next_gfn_state = (next_observed, next_values, next_cashed)

        # Convert structured array into dictionary
        # If we find that the training loop is too slow, we might want to
        # store the graphs tuples using replay.add directly by storing each
        # of its attributes separately (ugly solution, but saves performance)
        return {
            "observed": samples["observed"],
            "next_observed": samples["next_observed"],
            "graphs_tuple": to_graphs_tuple(self.full_cliques, gfn_state, self.K),
            "next_graphs_tuple": to_graphs_tuple(
                self.full_cliques, next_gfn_state, self.K
            ),
            "actions": samples["actions"],
            "done": samples["done"],
            "var_rewards": samples["var_rewards"],
            "value_rewards": samples["value_rewards"],
            "mask": samples["mask"],
            "next_mask": samples["next_mask"],
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
            "value_reward": np.zeros((1, 1), dtype=np.float_),
            "clique_reward": np.zeros((1, 1), dtype=np.float_),
            "mask": np.zeros(
                (
                    1,
                    self.num_variables,
                ),
                dtype=np.bool,
            ),
            "next_mask": np.zeros((1, self.num_variables), dtype=np.bool),
        }

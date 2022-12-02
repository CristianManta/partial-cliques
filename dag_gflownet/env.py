import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, h_dim, x_dim, K, graph, data):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.
        h_dim : int
            Number of latent variables.

        x_dim: int
            Number of low-level variables.

        graph: MarkovNetwork
            The ground truth UGM.

        data: dataframe
            data sampled from the ground truth UGM.
        """

        self._state = None
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.K = K
        self.num_variables = h_dim + x_dim
        self.graph = graph
        self.data = np.array(data)

        # TODO: Change this to the appropriate obs space
        observation_space = Dict(
            {
                "something": Box(low=0.0, high=1.0, shape=(1,), dtype=np.int_),
            }
        )
        action_space = Discrete(self.num_variables * self.K)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self):
        observed = np.zeros(self.num_variables, dtype=int)
        observed[self.h_dim :] = 1
        values = np.array([self.K] * self.num_variables)
        values[self.h_dim :] = self.data[
            np.random.randint(self.data.shape[0]),
            self.h_dim :,
        ]
        gfn_state = (
            observed,
            values,
            np.ones(self.num_variables, dtype=int),
        )
        self._state = {
            "gfn_state": gfn_state,
            "mask": np.ones(shape=(1, self.num_variables), dtype=int),
        }
        # mark x as observed and not eligible for sampling
        self._state["mask"][0, self.h_dim :] = 0
        return deepcopy(self._state)

    def step(self, actions):
        assert len(actions.shape) == 2
        assert actions.shape[0] == 1
        assert actions.shape[1] == 2
        self._state["gfn_state"][0][actions[0, 0]] = 1
        self._state["gfn_state"][1][actions[0, 0]] = actions[0, 1]
        self._state["mask"][0][actions[0, 0]] = 0

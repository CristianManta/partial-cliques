import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, h_dim, x_dim):
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
        """

        self._state = None
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.num_variables = h_dim + x_dim

        # TODO: Change this to contain (name, value) tuples
        # observation_space = Dict({
        #     'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int_),
        #     'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
        #     'num_edges': Discrete(max_edges),
        #     'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
        #     'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)
        # })
        # action_space = Discrete(self.num_variables ** 2 + 1) # TODO: Also need to change this
        observation_space, action_space = (None, None)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self):
        observed = np.zeros(self.num_variables, dtype=int)
        observed[self.h_dim :] = 1
        gfn_state = (
            observed,
            None,  # TODO: Need to know the values of x to fill this
            np.ones(self.num_variables, dtype=int),
            self.x_dim,
        )
        self._state = {
            "gfn_state": gfn_state,
            "mask": np.ones(shape=(1, self.num_variables), dtype=int),
        }
        return deepcopy(self._state)

    def step(self, actions):
        # TODO: Update current state given batch of actions
        raise NotImplementedError

import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, num_variables):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.
        num_variables : int
            Maximum number of latent variables that can be sampled
        """

        self._state = None
        self.num_variables = num_variables

        shape = (self.num_variables, self.num_variables)
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
        # TODO:
        # self._state = {
        #     'adjacency': np.zeros(shape, dtype=np.int_),
        #     'mask': 1 - self._closure_T,
        #     'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
        #     'score': np.zeros((self.num_envs,), dtype=np.float_),
        #     'order': np.full(shape, -1, dtype=np.int_)
        # }
        raise NotImplementedError

    def step(self, actions):
        # TODO: Update current state given batch of actions
        raise NotImplementedError

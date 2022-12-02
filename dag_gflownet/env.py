import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache
from dag_gflownet.utils.data import get_value_policy_reward, get_potential_fns


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, h_dim, x_dim, K, graph, full_cliques, clique_potentials, data):
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
        self.full_cliques = full_cliques
        self.clique_potentials = clique_potentials
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
            "unobserved_cliques": deepcopy(self.full_cliques)
        }
        # mark x as observed and not eligible for sampling
        self._state["mask"][0, self.h_dim :] = 0
        return deepcopy(self._state)

    def step(self, actions):
        # we use the convention that if actions[0][0] == -1, we terminate
        assert len(actions.shape) == 2
        assert actions.shape[0] == 1
        assert actions.shape[1] == 2
        obs_var = actions[0, 0]
        obs_value = actions[0, 1]
        if obs_var == -1:
            is_done = True
            
            mi_reward = 0. # TODO:
            value_reward = 0 # TODO: calculate partial reward by merging cliques
            
            return self.reset(), (mi_reward, value_reward), is_done
        is_done = False
        self._state["gfn_state"][0][obs_var] = 1
        self._state["gfn_state"][1][obs_var] = obs_value
        self._state["mask"][0][obs_var] = 0
        var_reward = 0. # TODO
        new_gfn_state, unobserved_cliques, value_reward = \
            get_value_policy_reward(self._state["gfn_state"],
                                    self._state["unobserved_cliques"],
                                    self.full_cliques,
                                    self.clique_potentials,
                                    self.K)
        self._state["gfn_state"] = new_gfn_state
        self._state['unobserved_cliques'] = unobserved_cliques
        return deepcopy(self._state), (var_reward, value_reward), is_done

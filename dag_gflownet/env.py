import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache
from dag_gflownet.utils.data import (
    get_value_policy_energy,
    get_potential_fns,
    get_clique_selection_mask,
)


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(
        self, num_envs, h_dim, x_dim, K, graph, full_cliques, clique_potentials, data
    ):
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
        # mark x as cashed
        gfn_state[2][self.h_dim :] = 0
        self._state = {
            "gfn_state": [gfn_state],
            "mask": np.ones(shape=(1, self.num_variables), dtype=int), # FIXME: assumption that batch size = 1
            "unobserved_cliques": [deepcopy(self.full_cliques)],
            "is_done": [False],
        }
        # mark x as observed and not eligible for sampling
        self._state["mask"][0, self.h_dim :] = 0
        return deepcopy(self._state)

    def step(self, actions):
        # we use the convention that if actions[0][0] == -1, we terminate
        assert len(actions.shape) == 2
        assert actions.shape[0] == 1
        assert actions.shape[1] == 2

        var_energies = []
        value_energies = []
        dones = []

        obs_var = actions[:, 0]
        obs_value = actions[:, 1]
        bsz = len(self._state["mask"]) # FIXME: See FIXME above

        assert np.all(
            (self._state["mask"][np.arange(bsz), obs_var] == 1)[actions[:, 0] != -1]
        )
        for i in range(bsz):
            assert np.all(
                (self._state["gfn_state"][i][0][obs_var] == 0)[actions[i, 0] != -1]
            )
            if actions[i, 0] == -1:
                continue
            self._state["gfn_state"][i][0][obs_var] = 1
            self._state["gfn_state"][i][1][obs_var] = obs_value
        var_energy = 0.0  # TODO

        for i in range(bsz):
            if obs_var == -1:
                self._state["is_done"][i] = True
                var_energies.append(0.0)
                (
                    new_gfn_state,
                    unobserved_cliques,
                    value_energy,
                ) = get_value_policy_energy(
                    self._state["gfn_state"][i],
                    self._state["unobserved_cliques"][i],
                    self.full_cliques,
                    self.clique_potentials,
                    self.K,
                    count_partial_cliques=True,
                    graph=self.graph,
                )
                value_energies.append(value_energy)
                self._state["unobserved_cliques"][i] = unobserved_cliques
                self._state["gfn_state"][i] = new_gfn_state
                dones.append([True])
                continue
            new_gfn_state, unobserved_cliques, value_energy = get_value_policy_energy(
                self._state["gfn_state"][i],
                self._state["unobserved_cliques"][i],
                self.full_cliques,
                self.clique_potentials,
                self.K,
            )

            self._state["unobserved_cliques"][i] = unobserved_cliques
            self._state["gfn_state"][i] = new_gfn_state

            self._state["mask"][i] = np.array(
                get_clique_selection_mask(
                    self._state["gfn_state"][i],
                    self._state["unobserved_cliques"][i],
                    self.K,
                )
            )
            var_energies.append(var_energy)
            value_energies.append(value_energy)
            dones.append([False])

        return deepcopy(self._state), (var_energies, value_energies), dones

import jax.numpy as jnp
import haiku as hk
import optax

from functools import partial
from jax import grad, random, jit

from collections import namedtuple

from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy
from dag_gflownet.utils.gflownet import (
    uniform_log_policy,
    detailed_balance_loss_free_energy_to_go,
)
from dag_gflownet.utils.jnp_utils import batch_random_choice
from dag_gflownet.utils.data import get_value_policy_reward

GFlowNetParameters = namedtuple("GFlowNetParameters", ["clique_model", "value_model"])


class DAGGFlowNet:
    """DAG-GFlowNet.

    Parameters
    ----------
    model : callable (optional)
        The neural network for the GFlowNet. The model must be a callable
        to feed into hk.transform, that takes a single adjacency matrix and
        a single mask as an input. Default to an architecture based on
        Linear Transformers.

    delta : float (default: 1.)
        The value of delta for the Huber loss used in the detailed balance
        loss (in place of the L2 loss) to avoid gradient explosion.
    """

    def __init__(self, clique_potentials, full_cliques, delta=1.0):

        clique_model = clique_policy
        value_model = value_policy

        self.clique_model = hk.without_apply_rng(hk.transform(clique_model))
        self.value_model = hk.without_apply_rng(hk.transform(value_model))
        self.delta = delta
        self.clique_potentials = clique_potentials
        self.full_cliques = full_cliques

        self._optimizer = None

    def loss(
        self, params, samples, x_dim, K
    ):  # TODO: Need to know how the samples look like
        # Then evaluate the models to obtain its log-probs

        # Example:
        log_probs_clique = self.clique_model.apply(
            params.clique_model, samples["graphs_tuple"], samples["mask"], x_dim, K
        )

        # OR
        log_probs_values, value_log_flows = self.value_model.apply(
            params.value_model, samples["graphs_tuple"], samples["mask"], x_dim, K
        )

        log_pf = log_probs_clique + log_probs_values
        # TODO: calculate PB for every sample by looking at how many variables are in them
        log_pb = None
        log_fetg_t = value_log_flows
        # TODO: calculate the fetg for s_{t+1}
        log_fetg_tp1 = None

        # ...
        partial_rewards = []
        for sample in samples:
            # TODO: extract gfn_state (a 3-tuple) and unobserved_cliques from sample
            gfn_state = None
            unobserved_cliques = None
            partial_rewards.append(
                get_value_policy_reward(
                    gfn_state,
                    unobserved_cliques,
                    self.full_cliques,
                    self.clique_potentials,
                )
            )
        partial_rewards = jnp.array(partial_rewards)

        return detailed_balance_loss_free_energy_to_go(
            log_fetg_t=log_fetg_t,
            log_fetg_tp1=log_fetg_tp1,
            log_pf=log_pf,
            log_pb=log_pb,
            partial_rewards=partial_rewards,
            delta=self.delta,
        )

    @partial(jit, static_argnums=(0, 5, 6))
    def act(self, params, key, observations, epsilon, x_dim, K):
        # masks = observations['mask'].astype(jnp.float32)
        # graphs = observations['graph']
        # batch_size = masks.shape[0]
        # key, subkey1, subkey2 = random.split(key, 3)

        # # Get the GFlowNet policy
        # log_pi = self.model.apply(params, graphs, masks)

        # # Get uniform policy
        # log_uniform = uniform_log_policy(masks)

        # # Mixture of GFlowNet policy and uniform policy
        # is_exploration = random.bernoulli(
        #     subkey1, p=1. - epsilon, shape=(batch_size, 1))
        # log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # # Sample actions
        # actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)

        # logs = {
        #     'is_exploration': is_exploration.astype(jnp.int32),
        # }
        # return (actions, key, logs)
        # TODO:
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 4, 5))
    def step(self, params, state, samples, x_dim, K):
        grads, logs = grad(self.loss, has_aux=True)(params, samples, x_dim, K)

        # Update the online params
        updates, state = self.optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask, x_dim, K):
        # Set the optimizer
        self._optimizer = optax.chain(optimizer, optax.zero_nans())

        # Initialize the models
        key1, key2 = random.split(key, 2)
        clique_params = self.clique_model.init(key1, graph, mask, x_dim, K)
        value_params = self.value_model.init(key2, graph, mask, x_dim, K)
        params = GFlowNetParameters(
            clique_model=clique_params, value_model=value_params
        )
        state = self.optimizer.init(params)
        return (params, state)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError(
                "The optimizer is not defined. To train the "
                "GFlowNet, you must call `DAGGFlowNet.init` first."
            )
        return self._optimizer

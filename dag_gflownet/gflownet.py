import jax.numpy as jnp
import jax
import haiku as hk
import optax
import numpy as np

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

    def __init__(self, x_dim, h_dim, delta=1.0):

        clique_model = clique_policy
        value_model = value_policy

        self.clique_model = hk.without_apply_rng(hk.transform(clique_model))
        self.value_model = hk.without_apply_rng(hk.transform(value_model))
        self.delta = delta
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.N = x_dim + h_dim

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

        log_pf = log_probs_values
        log_pb = 1 / (samples["observed"].sum(axis=-1) - self.x_dim)
        log_fetg_t = value_log_flows
        _, log_fetg_tp1 = self.value_model.apply(
            params.value_model,
            samples["next_graphs_tuple"],
            samples["next_mask"],
            x_dim,
            K,
        )
        partial_rewards = samples[
            "reward"
        ]  # TODO: I think that here you mean value_rewards

        return detailed_balance_loss_free_energy_to_go(
            log_fetg_t=log_fetg_t,
            log_fetg_tp1=log_fetg_tp1,
            log_pf=log_pf,
            log_pb=log_pb,
            partial_rewards=partial_rewards,
            delta=self.delta,
        )

    # @partial(jit, static_argnums=(0, 5, 6))
    def act(self, params, key, observations, epsilon, x_dim, K):

        graphs = observations["graphs_tuple"]
        masks = observations["mask"].astype(jnp.float32)
        batch_size = masks.shape[0]
        key, subkey1, subkey2 = random.split(key, 3)

        # First get the clique policy
        log_probs_clique = self.clique_model.apply(
            params.clique_model, graphs, masks, x_dim, K
        )

        # Get uniform policy
        # log_uniform = uniform_log_policy(masks)

        # Mixture of GFlowNet policy and uniform policy
        # is_exploration = random.bernoulli(
        #     subkey1, p=1. - epsilon, shape=(batch_size, 1))
        # log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        # clique_policy_actions = batch_random_choice(subkey2, jnp.exp(log_probs_clique), masks)
        clique_actions = jax.random.categorical(
            subkey1, log_probs_clique[0] / 999
        )  # a single integer between 0 and h_dim

        if clique_actions == self.h_dim:
            # we are done!
            logs = {
                "is_exploration": None,  # TODO:
            }

            actions = np.array([-1, -1])
            return actions, key, logs

        graphs.values.nodes[clique_actions] = self.N + K + 1

        # use the value GFN to sample a value for the variable we just observed
        log_probs_value, log_flow = self.value_model.apply(
            params.value_model, graphs, masks, x_dim, K
        )

        sampled_value = jax.random.categorical(subkey1, log_probs_value)

        actions = np.array([clique_actions, sampled_value])

        logs = {
            "is_exploration": None,
        }
        return (actions, key, logs)

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

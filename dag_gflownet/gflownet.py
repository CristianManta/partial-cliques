import jax.numpy as jnp
import haiku as hk
import optax

from functools import partial
from jax import grad, random, jit

from collections import namedtuple

from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy
from dag_gflownet.utils.gflownet import uniform_log_policy, detailed_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice

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

    def __init__(self, delta=1.0):

        clique_model = clique_policy
        value_model = value_policy

        self.clique_model = hk.without_apply_rng(hk.transform(clique_model))
        self.value_model = hk.without_apply_rng(hk.transform(value_model))
        self.delta = delta

        self._optimizer = None

    def loss(self, params, samples):  # TODO: Need to know how the samples look like
        # Then evaluate the models to obtain its log-probs

        # Example:
        log_probs_clique = self.clique_model.apply(
            params.clique_model, samples["graphs_tuple"], samples["mask"]
        )

        # OR
        log_probs_values, value_log_flows = self.value_model.apply(
            params.value_model, samples["graphs_tuple"], samples["mask"]
        )

        # ...

        # TODO: Add custom loss here
        # return detailed_balance_loss(
        #     log_pi_t,
        #     log_pi_tp1,
        #     samples['actions'],
        #     samples['delta_scores'],
        #     samples['num_edges'],
        #     delta=self.delta
        # )
        # TODO:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon):
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

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples):
        grads, logs = grad(self.loss, has_aux=True)(params, samples)

        # Update the online params
        updates, state = self.optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask):
        # Set the optimizer
        self._optimizer = optax.chain(optimizer, optax.zero_nans())

        # Initialize the models
        key1, key2 = random.split(key, 2)
        clique_params = self.clique_model.init(key1, graph, mask)
        value_params = self.value_model.init(key2, graph, mask)
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

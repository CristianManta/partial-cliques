import jax.numpy as jnp
import jax
import haiku as hk
import optax
import numpy as np

from functools import partial
from jax import grad, random, jit, nn

from collections import namedtuple

from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy, value_policy_MLP
from dag_gflownet.nets.transformer.gflownet import value_policy_transformer
from dag_gflownet.utils.gflownet import (
    uniform_log_policy,
    detailed_balance_loss_free_energy_to_go,
)
from dag_gflownet.utils.jnp_utils import batch_random_choice
from dag_gflownet.utils.data import get_value_policy_energy
from dag_gflownet.utils.jraph_utils import Graph

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
        value_model = value_policy_transformer

        self.clique_model = hk.without_apply_rng(hk.transform(clique_model))
        self.value_model = hk.transform(value_model)
        self.delta = delta
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.N = x_dim + h_dim

        self._optimizer = None

    def loss(
        self, params, samples, x_dim, K, forward_key
    ):  # TODO: Need to know how the samples look like
        # Then evaluate the models to obtain its log-probs

        # Example:
        """
        log_probs_clique = self.clique_model.apply(
            params.clique_model,
            samples["graphs_tuple"],
            samples["mask"],
            x_dim,
            K,
        )
        """
        # OR
        # calculate batch size

        forward_key, _ = jax.random.split(forward_key, 2)

        bsz = samples["observed"].shape[0]
        logits_value, value_log_flows = self.value_model.apply(
            params.value_model,
            forward_key,
            samples["graphs_tuple"],
            samples["mask"],
            x_dim,
            K,
        )

        log_pf = nn.log_softmax(logits_value)[jnp.arange(bsz), samples["actions"][:, 1]]
        log_pb = jnp.zeros_like(log_pf)
        # log_pb = jnp.where(
        #     samples["dones"],
        #     0,
        #     jnp.log(
        #         1
        #         / (
        #             samples["next_observed"].sum(axis=-1, keepdims=True) # FIXME: I don't understand how does that relate to the formula from Yoshua's notion
        #             - self.x_dim
        #             + 1e-8
        #         )
        #     ),
        # ).squeeze(axis=-1)
        """
        if (
            jnp.any(jnp.isnan(log_pb))
            or jnp.any(jnp.isinf(log_pb))
            or jnp.any(log_pb > 10000)
        ):
            raise NotImplementedError
        """
        log_fetg_t = value_log_flows
        _, log_fetg_tp1 = self.value_model.apply(
            params.value_model,
            forward_key,
            samples["next_graphs_tuple"],
            samples["next_mask"],
            x_dim,
            K,
        )

        value_energies = samples["value_energies"]
        fetg_tp1_done = samples["next_observed"].all(axis=-1)
        log_fetg_tp1 = jnp.where(fetg_tp1_done, 0, log_fetg_tp1)
        unfiltered_loss, logs = detailed_balance_loss_free_energy_to_go(
            log_fetg_t=log_fetg_t,
            log_fetg_tp1=log_fetg_tp1,
            log_pf=log_pf,
            log_pb=log_pb,
            partial_energies=value_energies,
            delta=self.delta,
            reduction="none",
        )
        loss = jnp.where(
            samples["dones"].squeeze(axis=-1), 0, unfiltered_loss
        ).sum() / (jnp.sum(~samples["dones"]) + 1e-18)
        logs["loss"] = loss
        logs["forward_key"] = forward_key
        return (loss, logs)

    # @partial(jit, static_argnums=(0, 5, 6))
    def act(self, params, key, observations, epsilon, x_dim, K, temperature=1.0):

        graphs = observations["graphs_tuple"]
        masks = observations["mask"].astype(jnp.float32)
        batch_size = masks.shape[0]
        key, subkey1, forward_key = random.split(key, 3)

        # First get the clique policy
        log_probs_clique = self.clique_model.apply(
            params.clique_model, graphs, masks, x_dim, K, sampling_method=2
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
            subkey1, log_probs_clique / 999
        )  # a single integer between 0 and h_dim

        """
        if clique_actions == self.h_dim:
            # we are done!
            logs = {
                "is_exploration": None,  # TODO:
            }

            actions = np.array([-1, -1])
            return actions, key, logs
        """
        for i in range(batch_size):
            new_nodes = graphs.values.nodes.at[i * self.N + clique_actions[i]].set(
                self.N + K + 1
            )
            graphs = Graph(
                structure=graphs.structure,
                values=graphs.values._replace(nodes=new_nodes),
            )

        # use the value GFN to sample a value for the variable we just observed
        logits_value, log_flow = self.value_model.apply(
            params.value_model, forward_key, graphs, masks, x_dim, K
        )

        sampled_value = jax.random.categorical(subkey1, logits_value / temperature)

        logpf = nn.log_softmax(logits_value)[jnp.arange(batch_size), sampled_value]

        actions = jnp.stack([clique_actions, sampled_value], axis=-1)
        actions = jnp.where(
            clique_actions == self.h_dim, -jnp.ones_like(actions), actions
        )
        logs = {
            "is_exploration": None,
            "logpf": logpf,
        }
        return (actions, key, logs)

    def compute_data_log_likelihood(
        self, params, init_observation, x_dim, K, true_partition_fn, forward_key
    ):
        forward_key, _ = jax.random.split(forward_key, 2)
        graphs = init_observation["graphs_tuple"]
        masks = init_observation["mask"].astype(jnp.float32)
        _, log_flow = self.value_model.apply(
            params.value_model, forward_key, graphs, masks, x_dim, K
        )

        log_true_partition_fn = jnp.log(true_partition_fn)
        log_p_hat = log_flow - log_true_partition_fn
        return log_p_hat, forward_key

    def compute_reverse_kl(self, full_observations, full_cliques, traj_pf, ugm_model):
        # compute the reverse KL(GFN || GT)
        assert full_observations.shape[0] == traj_pf.shape[0]
        factors = ugm_model.get_factors()
        # for every sample, compute the likelihood under the ugm_model
        kl_terms = []
        for i in range(full_observations.shape[0]):
            # for every factor, compute the associated clique potential
            total_log_potential = 0
            for c_ind, factor in enumerate(factors):
                total_log_potential += np.log(
                    factor.values[
                        tuple(
                            full_observations[i][
                                np.array(sorted(list(full_cliques[c_ind])))
                            ]
                        )
                    ]
                )
            # store the KL term
            kl_terms.append(
                traj_pf[i]
                - total_log_potential
                + np.log(ugm_model.get_partition_function())
            )
        return jnp.mean(jnp.array(kl_terms))

    @partial(jit, static_argnums=(0, 4, 5))
    def step(self, params, state, samples, x_dim, K, forward_key):
        grads, logs = grad(self.loss, has_aux=True)(
            params, samples, x_dim, K, forward_key
        )
        forward_key = logs["forward_key"]

        # Update the online params
        updates, state = self.optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        return (params, state, logs, forward_key)

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

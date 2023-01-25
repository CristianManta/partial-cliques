import jax.numpy as jnp
import jax
import haiku as hk
import optax
import numpy as np

from copy import deepcopy

from functools import partial
from jax import grad, random, jit, nn

from collections import namedtuple

from dag_gflownet.nets.gnn.gflownet import clique_policy, value_policy, value_policy_MLP
from dag_gflownet.nets.transformer.gflownet import (
    value_policy_transformer,
    clique_policy_transformer,
    random_clique_policy,
)
from dag_gflownet.utils.gflownet import (
    uniform_log_policy,
    detailed_balance_loss_free_energy_to_go,
    mask_logits,
    MASKED_VALUE,
)
from dag_gflownet.utils.jnp_utils import batch_random_choice
from dag_gflownet.utils.data import get_value_policy_energy, find_incomplete_clique
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

    def __init__(
        self,
        x_dim,
        h_dim,
        delta=1.0,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        key_size=32,
        dropout_rate=0.0,
        pb="uniform",
        full_cliques=None,
    ):

        self.delta = delta
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.N = x_dim + h_dim

        self._optimizer = None

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.key_size = key_size
        self.dropout_rate = dropout_rate
        self.pb = pb
        self.full_cliques = full_cliques

        if self.pb == "uniform":
            clique_model = random_clique_policy
            self.clique_model = hk.transform(clique_model)
        elif self.pb == "deterministic":
            clique_model = clique_policy
            self.clique_model = hk.without_apply_rng(hk.transform(clique_model))

        value_model = value_policy_transformer

        self.value_model = hk.transform(value_model)

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
        forward_key_clique, _ = jax.random.split(forward_key, 2)

        bsz = samples["observed"].shape[0]
        logits_value, value_log_flows = self.value_model.apply(
            params.value_model,
            forward_key,
            samples["values"],
            samples["mask"],
            x_dim,
            K,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            key_size=self.key_size,
            dropout_rate=self.dropout_rate,
        )

        if self.pb == "uniform":
            log_pf_clique = jnp.zeros((bsz,))

        elif self.pb == "deterministic":
            log_pf_clique = jnp.zeros((bsz,))

        log_pf_value = nn.log_softmax(logits_value)[
            jnp.arange(bsz), samples["actions"][:, 1]
        ]

        log_pf = log_pf_value + log_pf_clique

        if self.pb == "uniform":
            log_pb = jnp.zeros_like(log_pf)

        elif self.pb == "deterministic":
            log_pb = jnp.zeros_like(log_pf)
        elif self.pb == "learnable":
            raise NotImplementedError()  # TODO
        else:
            raise ValueError("Invalid pb choice.")

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
            samples["next_values"],
            samples["next_mask"],
            x_dim,
            K,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            key_size=self.key_size,
            dropout_rate=self.dropout_rate,
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
        if self.pb == "uniform":

            logits_clique = self.clique_model.apply(
                params.clique_model,
                key,
                observations["gfn_state"][0][1].reshape(1, -1),
                masks,
                x_dim,
                K,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                key_size=self.key_size,
                dropout_rate=self.dropout_rate,
            )

            logits_clique = mask_logits(logits_clique, masks[:, : self.h_dim])
            clique_actions = jax.random.categorical(subkey1, logits_clique)  # / 999

            clique_actions = jnp.where(
                jnp.all(logits_clique == MASKED_VALUE, axis=-1),
                self.h_dim,
                clique_actions,
            )  # h_dim means stop action.
            # Only output the stop action when there is no other available action

        elif self.pb == "deterministic":

            log_probs_clique = self.clique_model.apply(
                params.clique_model, graphs, masks, x_dim, K, sampling_method=2
            )

            clique_actions = jax.random.categorical(
                subkey1, log_probs_clique / 999
            )  # a single integer between 0 and h_dim

        elif self.pb == "learnable":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid pb choice")

        # Get uniform policy
        # log_uniform = uniform_log_policy(masks)

        # Mixture of GFlowNet policy and uniform policy
        # is_exploration = random.bernoulli(
        #     subkey1, p=1. - epsilon, shape=(batch_size, 1))
        # log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        # clique_policy_actions = batch_random_choice(subkey2, jnp.exp(log_probs_clique), masks)

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
            """
            new_nodes = graphs.values.nodes.at[i * self.N + clique_actions[i]].set(
                self.N + K + 1
            )
            graphs = Graph(
                structure=graphs.structure,
                values=graphs.values._replace(nodes=new_nodes),
            )
            """
            assert clique_actions[i] <= self.h_dim
            if clique_actions[i] < self.h_dim:
                observations["gfn_state"][i][1][clique_actions[i]] = K + 1

        # use the value GFN to sample a value for the variable we just observed
        logits_value, log_flow = self.value_model.apply(
            params.value_model,
            forward_key,
            observations["gfn_state"][0][1].reshape(1, -1),
            masks,
            x_dim,
            K,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            key_size=self.key_size,
            dropout_rate=self.dropout_rate,
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
            "logz": log_flow,
        }
        return (actions, key, logs)

    def compute_data_log_likelihood(
        self, params, init_observation, x_dim, K, true_partition_fn, forward_key
    ):
        forward_key, _ = jax.random.split(forward_key, 2)
        graphs = init_observation["graphs_tuple"]
        masks = init_observation["mask"].astype(jnp.float32)
        _, log_flow = self.value_model.apply(
            params.value_model,
            forward_key,
            init_observation["gfn_state"][0][1].reshape(1, -1),
            masks,
            x_dim,
            K,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            key_size=self.key_size,
            dropout_rate=self.dropout_rate,
        )

        log_true_partition_fn = jnp.log(true_partition_fn)
        log_p_hat = log_flow - log_true_partition_fn
        return log_p_hat, forward_key

    def compute_reverse_kl(
        self, full_observations, full_cliques, traj_pf, log_marginal, ugm_model
    ):
        # compute the reverse KL(GFN || GT)
        assert full_observations.shape[0] == traj_pf.shape[0]
        assert full_observations.shape[0] == log_marginal.shape[0]
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
            kl_terms.append(traj_pf[i] + log_marginal - total_log_potential)
            # + np.log(ugm_model.get_partition_function())
        return jnp.mean(jnp.array(kl_terms))

    # @partial(jit, static_argnums=(0, 4, 5))
    def step(self, params, state, samples, x_dim, K, forward_key):
        grads, logs = grad(self.loss, has_aux=True)(
            params, samples, x_dim, K, forward_key
        )
        forward_key = logs["forward_key"]

        # Update the online params
        updates, state = self.optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        return (params, state, logs, forward_key)

    def init(self, key, optimizer, graph, values, mask, x_dim, K):
        # Set the optimizer
        self._optimizer = optax.chain(optimizer, optax.zero_nans())

        # Initialize the models
        key1, key2 = random.split(key, 2)

        if self.pb == "uniform":

            clique_params = self.clique_model.init(
                key1,
                values,
                mask,
                x_dim,
                K,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                key_size=self.key_size,
                dropout_rate=self.dropout_rate,
            )

        elif self.pb == "deterministic":

            clique_params = self.clique_model.init(
                key1, graph, mask, x_dim, K, sampling_method=2
            )

        elif self.pb == "learnable":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid pb choice")

        value_params = self.value_model.init(
            key2,
            values,
            mask,
            x_dim,
            K,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            key_size=self.key_size,
            dropout_rate=self.dropout_rate,
        )
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

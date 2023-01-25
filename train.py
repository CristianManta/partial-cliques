import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import jax
import wandb
import os

from tqdm import trange
from numpy.random import default_rng
from pgmpy.factors import factor_sum_product

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.data import get_data, get_potential_fns, get_energy_fns
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import (
    expected_shd,
    expected_edges,
    threshold_metrics,
    get_log_features,
)
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils import io
from dag_gflownet.utils.wandb_utils import (
    slurm_infos,
    table_from_dict,
    scatter_from_dicts,
)
from dag_gflownet.utils.exhaustive import (
    get_full_posterior,
    get_edge_log_features,
    get_path_log_features,
    get_markov_blanket_log_features,
)


def main(args):
    if not args.off_wandb:
        wandb.init(
            project="partial-cliques",
            group="energy-based",
            tags=["gnn"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)
        wandb.run.summary.update(slurm_infos())

    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Generate the ground truth data
    graph, data, _ = get_data("random_latent_graph", args, rng=rng)
    train_data, eval_data = data
    # latent_data, obs_data = data
    (
        true_ugm,
        full_cliques,
        factors,
    ) = graph

    np.save(f"train_data_h_dim_{args.h_dim}", train_data.to_numpy())
    np.save(f"eval_data_h_dim_{args.h_dim}", eval_data.to_numpy())

    true_partition_fn = true_ugm.get_partition_function()
    obs_nodes = ["x" + str(i) for i in range(args.x_dim)]
    x_factors_values = factor_sum_product(
        output_vars=obs_nodes, factors=true_ugm.factors
    ).values
    # x_factors_values[np.array(eval_data[obs_nodes])[:,0], np.array(eval_data[obs_nodes])[:,1]]
    indexing = [np.array(eval_data[obs_nodes])[:, i] for i in range(args.x_dim)]
    eval_unnormalized_probs = x_factors_values[tuple(indexing)]
    log_p_x_eval = np.log(eval_unnormalized_probs) - np.log(true_partition_fn)
    # instead of using sum-product to get the unormalized probabilities, use the factors directly to get the energies
    # clique_potentials = get_potential_fns(true_ugm, full_cliques)
    clique_potentials = factors
    # clique_energies = get_energy_fns(true_ugm, full_cliques)

    # Create the environment
    # TODO:
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        h_dim=args.h_dim,
        x_dim=args.x_dim,
        clique_potentials=clique_potentials,
        full_cliques=full_cliques,
        K=args.K,
        graph=true_ugm,
        data=train_data,
        structure=args.latent_structure,
        pb=args.pb,
    )

    eval_env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        h_dim=args.h_dim,
        x_dim=args.x_dim,
        clique_potentials=clique_potentials,
        full_cliques=full_cliques,
        K=args.K,
        graph=true_ugm,
        data=eval_data,
        structure=args.latent_structure,
        pb=args.pb,
    )

    # Create the replay buffer
    replay = ReplayBuffer(  # TODO: Implement replay buffer
        args.replay_capacity,
        full_cliques,
        args.K,
        num_variables=args.h_dim + args.x_dim,
        x_dim=args.x_dim,
    )

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        delta=args.delta,
        x_dim=args.x_dim,
        h_dim=args.h_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        key_size=args.key_size,
        dropout_rate=args.dropout_rate,
        pb=args.pb,
        full_cliques=full_cliques,
    )
    if args.optimizer == "adam":
        optimizer = optax.adam(args.lr)
    elif args.optimizer == "sgd":
        optimizer = optax.sgd(args.lr)
    else:
        raise ValueError("Optimizer name is invalid.")

    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy["graph"],
        replay.dummy["values"],
        replay.dummy["mask"],
        args.x_dim,
        args.K,
    )
    exploration_schedule = jax.jit(
        optax.linear_schedule(
            init_value=jnp.array(0.0),
            end_value=jnp.array(1.0 - args.min_exploration),
            transition_steps=args.num_iterations // 2,
            transition_begin=args.prefill,
        )
    )

    # Training loop
    observations = env.reset()  # For the training code (this will get updated)
    init_eval_observation = eval_env.reset()  # For the evaluation code
    init_eval_observation["graphs_tuple"] = to_graphs_tuple(
        full_cliques, init_eval_observation["gfn_state"], args.K, args.x_dim
    )

    traj_length = 0
    with trange(args.prefill + args.num_iterations, desc="Training") as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations["graphs_tuple"] = to_graphs_tuple(
                full_cliques, observations["gfn_state"], args.K, args.x_dim
            )
            actions, key, logs = gflownet.act(
                params, key, observations, epsilon, args.x_dim, args.K, temperature=2.0
            )
            next_observations, energies, dones = env.step(actions)
            replay.add(
                observations,
                actions,
                logs["is_exploration"],
                next_observations,
                energies,
                dones,
            )

            if dones[0][0] or traj_length >= args.max_traj_length:
                observations = env.reset()
                traj_length = 0
            else:
                observations = next_observations
                traj_length += 1

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs, key = gflownet.step(
                    params, state, samples, args.x_dim, args.K, key
                )

                # Evaluation: compute log p(x_eval)
                log_p_hat_x_eval, key = gflownet.compute_data_log_likelihood(
                    params,
                    init_eval_observation,
                    args.x_dim,
                    args.K,
                    true_partition_fn,
                    key,
                )

                train_steps = iteration - args.prefill
                if not args.off_wandb:
                    if (train_steps + 1) % (args.log_every * 10) == 0:
                        wandb.log(
                            {
                                "replay/is_exploration": np.mean(
                                    replay.transitions["is_exploration"]
                                )
                            },
                            commit=False,
                        )
                    if (train_steps + 1) % args.log_every == 0:
                        wandb.log(
                            {
                                "step": train_steps,
                                "loss": logs["loss"],
                                "log_p_hat_x_eval": log_p_hat_x_eval[0],
                                "log_p_x_eval": log_p_x_eval.mean(),
                                "replay/size": len(replay),
                                "epsilon": epsilon,
                                "error/mean": jnp.abs(logs["error"]).mean(),
                                "error/max": jnp.abs(logs["error"]).max(),
                            }
                        )
                pbar.set_postfix(
                    loss=f"{logs['loss']:.2f}",
                    epsilon=f"{epsilon:.2f}",
                    MLL=f"{log_p_hat_x_eval[0]:.2f}",
                )

                if (train_steps) % args.evaluate_every == 0:
                    # evaluete the GFN by sampling complete trajectories
                    eval_full_trajectories = []
                    eval_logpf = []
                    eval_logz = []
                    eval_log_marginal = []
                    eval_obs = eval_env.reset()
                    for _ in range(100):
                        logpf = 0.0
                        logz = 0.0
                        eval_obs["graphs_tuple"] = to_graphs_tuple(
                            full_cliques, eval_obs["gfn_state"], args.K, args.x_dim
                        )
                        actions, key, logs = gflownet.act(
                            params,
                            key,
                            eval_obs,
                            epsilon,
                            args.x_dim,
                            args.K,
                            temperature=1.0,
                        )
                        eval_obs, energies, dones = eval_env.step(actions)
                        logpf += logs["logpf"]
                        logz += logs["logz"]

                        if dones[0][0]:
                            eval_full_trajectories.append(eval_obs["gfn_state"][0][1])
                            eval_logpf.append(logpf)
                            eval_logz.append(logz)
                            eval_log_marginal.append(
                                np.log(
                                    x_factors_values[
                                        tuple(
                                            [
                                                eval_obs["gfn_state"][0][1][i]
                                                for i in range(args.x_dim)
                                            ]
                                        )
                                    ]
                                )
                            )
                            eval_obs = eval_env.reset()
                            logpf = 0.0
                    # calculate and print reverse KL
                    reverse_kl = gflownet.compute_reverse_kl(
                        full_observations=jnp.stack(eval_full_trajectories, axis=0),
                        full_cliques=full_cliques,
                        traj_pf=jnp.array(eval_logpf),
                        log_marginal=jnp.array(eval_log_marginal),
                        ugm_model=true_ugm,
                    )
                    print(f"Reverse KL: {reverse_kl}")
                    if not args.off_wandb:
                        wandb.log(
                            {
                                "Reverse KL": reverse_kl,
                            }
                        )

    # Sample from the learned policy
    # TODO:
    # learned_graphs = sample_from(
    #     gflownet,
    #     params,
    #     env,
    #     key,
    #     num_samples=args.num_learned_samples,
    # )

    # Compute the metrics
    # TODO: This could serve as an inspiration for our evaluation as well
    # ground_truth = nx.to_numpy_array(graph, weight=None)
    # wandb.run.summary.update({
    #     'metrics/shd/mean': expected_shd(posterior, ground_truth),
    #     'metrics/edges/mean': expected_edges(posterior),
    #     'metrics/thresholds': threshold_metrics(posterior, ground_truth)
    # })

    # if (args.graph in ['erdos_renyi_lingauss']) and (args.num_variables < 6):
    #     log_features = get_log_features(posterior, data.columns)
    #     full_posterior = get_full_posterior(data, scorer, verbose=True)
    #     full_posterior.save(os.path.join(wandb.run.dir, 'posterior_full.npz'))
    #     wandb.save('posterior_full.npz', policy='now')

    #     full_edge_log_features = get_edge_log_features(full_posterior)
    #     full_path_log_features = get_path_log_features(full_posterior)
    #     full_markov_log_features = get_markov_blanket_log_features(full_posterior)

    #     wandb.log({
    #         'posterior/scatter/edge': scatter_from_dicts('full', full_edge_log_features,
    #             'estimate', log_features.edge, transform=np.exp, title='Edge features'),
    #         'posterior/scatter/path': scatter_from_dicts('full', full_path_log_features,
    #             'estimate', log_features.path, transform=np.exp, title='Path features'),
    #         'posterior/scatter/markov_blanket': scatter_from_dicts('full', full_markov_log_features,
    #             'estimate', log_features.markov_blanket, transform=np.exp, title='Markov blanket features')
    #     })

    # TODO: Save model, data & results
    # data.to_csv(os.path.join(wandb.run.dir, 'data.csv'))
    # wandb.save('data.csv', policy='now')

    # with open(os.path.join(wandb.run.dir, 'graph.pkl'), 'wb') as f:
    #     pickle.dump(graph, f)
    # wandb.save('graph.pkl', policy='now')

    # io.save(os.path.join(wandb.run.dir, 'model.npz'), params=params)
    # wandb.save('model.npz', policy='now')

    # replay.save(os.path.join(wandb.run.dir,  'replay_buffer.npz'))
    # wandb.save('replay_buffer.npz', policy='now')

    # np.save(os.path.join(wandb.run.dir, 'posterior.npy'), posterior)
    # wandb.save('posterior.npy', policy='now')


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description="DAG-GFlowNet for Strucure Learning.")

    # Environment
    environment = parser.add_argument_group("Environment")
    environment.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: %(default)s)",
    )

    # Optimization
    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: %(default)s)"
    )
    optimization.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Value of delta for Huber loss (default: %(default)s)",
    )
    optimization.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the number of elements to sample from the replay buffer (default: %(default)s)",
    )
    optimization.add_argument(
        "--num_iterations",
        type=int,
        default=100_000,
        help="Number of iterations (default: %(default)s)",
    )

    optimization.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer name. Choices: sgd or adam (default: %(default)s)",
    )

    # Replay buffer
    replay = parser.add_argument_group("Replay Buffer")
    replay.add_argument(
        "--replay_capacity",
        type=int,
        default=100_000,
        help="Capacity of the replay buffer (default: %(default)s)",
    )
    replay.add_argument(
        "--prefill",
        type=int,
        default=1000,
        help="Number of iterations with a random policy to prefill "
        "the replay buffer (default: %(default)s)",
    )
    replay.add_argument(
        "--max_traj_length",
        type=int,
        default=1000,
        help="Maximal length of the trajectory to include in "
        "the replay buffer (default: %(default)s)",
    )

    # Exploration
    exploration = parser.add_argument_group("Exploration")
    exploration.add_argument(
        "--min_exploration",
        type=float,
        default=0.1,
        help="Minimum value of epsilon-exploration (default: %(default)s)",
    )

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--num_learned_samples",
        type=int,
        default=1000,
        help="How many samples to draw from the learned GFN policy for evaluation? (default: %(default)s)",
    )
    misc.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: %(default)s)"
    )
    misc.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Frequency for logging (default: %(default)s)",
    )
    misc.add_argument(
        "--evaluate_every",
        type=int,
        default=100,
        help="Frequency for evaluating (default: %(default)s)",
    )
    misc.add_argument(
        "--off_wandb",
        action="store_true",
        default=False,
        help="Whether to use Wandb for logs (default: %(default)s)",
    )

    misc.add_argument(
        "--pb",
        type=str,
        default="uniform",
        help=(
            "backwards probability parametrization. "
            "Choices: uniform, learnable or deterministic. (default: %(default)s). "
            "If the deterministic option is chosen, log_pb will be 0 and "
            "the clique policy will simply sample all the latent variables "
            "in increasing order of their index."
        ),
    )

    # Graph
    graph_args = parser.add_argument_group("Graph")

    graph_args.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="How many samples to draw for the ground truth observations x?",
    )
    graph_args.add_argument(
        "--num_eval_samples",
        type=int,
        required=True,
        help="How many evalaution samples to draw for the ground truth observations x?",
    )
    graph_args.add_argument(
        "--x_dim", type=int, required=True, help="The number of observations variables?"
    )
    graph_args.add_argument(
        "--h_dim", type=int, required=True, help="The number of latent variables?"
    )
    graph_args.add_argument(
        "--K",
        type=int,
        required=True,
        help="The number of discrete values that the variables can take?",
    )

    graph_args.add_argument(
        "--latent_structure",
        type=str,
        default="random",
        help="type of graph. For now, choices are random or random_chain_graph_c3 (default: %(default)s)",
    )

    transformer_args = parser.add_argument_group("Transformer")
    transformer_args.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Number of dimensions of the embeddings sent to the transformer as input",
    )
    transformer_args.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads for the transformer",
    )
    transformer_args.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers for the transformer",
    )
    transformer_args.add_argument(
        "--key_size",
        type=int,
        default=32,
        help="Dimension of the key for the multi head attention mechanism",
    )
    transformer_args.add_argument(
        "--dropout_rate", type=float, default=0.0, help="Dropout rate."
    )

    args = parser.parse_args()

    main(args)

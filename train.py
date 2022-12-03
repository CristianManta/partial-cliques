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

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.data import get_data, get_potential_fns
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
    # TODO:
    graph, data, _ = get_data("random_latent_graph", args, rng=rng)
    # latent_data, obs_data = data
    true_ugm, full_cliques = graph
    clique_potentials = get_potential_fns(true_ugm, full_cliques)

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
        data=data,
    )

    # Create the replay buffer
    replay = ReplayBuffer(  # TODO: Implement replay buffer
        args.replay_capacity,
        full_cliques,
        args.K,
        num_variables=args.h_dim + args.x_dim,
    )

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(delta=args.delta, x_dim=args.x_dim, h_dim=args.h_dim)
    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy["graph"],
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
    observations = env.reset()
    with trange(args.prefill + args.num_iterations, desc="Training") as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations["graphs_tuple"] = to_graphs_tuple(
                full_cliques, observations["gfn_state"], args.K
            )
            actions, key, logs = gflownet.act(
                params, key, observations, epsilon, args.x_dim, args.K
            )  # TODO:
            next_observations, rewards, dones = env.step(
                np.asarray(actions)[np.newaxis, ...]
            )
            replay.add(  # TODO:
                observations,
                actions,
                logs["is_exploration"],
                next_observations,
                rewards,
                dones,
            )

            if dones:
                observations = env.reset()
            else:
                observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(
                    params, state, samples, args.x_dim, args.K
                )

                train_steps = iteration - args.prefill
                if not args.off_wandb:
                    if (train_steps + 1) % (args.log_every * 10) == 0:
                        wandb.log(
                            {
                                "replay/num_edges": wandb.Histogram(
                                    replay.transitions["num_edges"]
                                ),  # TODO: add more appropriate logs
                                "replay/is_exploration": np.mean(
                                    replay.transitions["is_exploration"]
                                ),
                            },
                            commit=False,
                        )
                    if (train_steps + 1) % args.log_every == 0:
                        wandb.log(
                            {
                                "step": train_steps,
                                "loss": logs["loss"],
                                "replay/size": len(replay),
                                "epsilon": epsilon,
                                "error/mean": jnp.abs(logs["error"]).mean(),
                                "error/max": jnp.abs(logs["error"]).max(),
                            }
                        )
                pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

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
        "--off_wandb",
        action="store_true",
        default=False,
        help="Whether to use Wandb for logs (default: %(default)s)",
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

    args = parser.parse_args()

    main(args)

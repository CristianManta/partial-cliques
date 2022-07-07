import jax.numpy as jnp
import haiku as hk
import jraph

from jax import lax

from dag_gflownet.utils.gflownet import log_policy


def gflownet(graphs, masks):
    batch_size, num_variables = masks.shape[:2]

    # Embedding of the nodes & edges
    node_embeddings = hk.Embed(num_variables, embed_dim=128)
    edge_embeddings = hk.Embed(2, embed_dim=128)

    graphs = graphs._replace(
        nodes=node_embeddings(graphs.nodes),
        edges=edge_embeddings(graphs.edges),
        globals=jnp.zeros((graphs.n_node.shape[0], 1)),
    )

    # Define graph network updates
    @jraph.concatenated_args
    def update_node_fn(features):
        return hk.nets.MLP([128, 128], name='node')(features)

    @jraph.concatenated_args
    def update_edge_fn(features):
        return hk.nets.MLP([128, 128], name='edge')(features)

    @jraph.concatenated_args
    def update_global_fn(features):
        return hk.nets.MLP([128, 128], name='global')(features)

    graph_net = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
    )
    features = graph_net(graphs)

    node_features = features.nodes[:batch_size * num_variables]

    senders = hk.nets.MLP([128, 128], name='senders')(node_features)
    senders = senders.reshape(batch_size, num_variables, -1)

    receivers = hk.nets.MLP([128, 128], name='receivers')(node_features)
    receivers = receivers.reshape(batch_size, num_variables, -1)

    logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
    stop = hk.nets.MLP([128, 1], name='stop')(features.globals)

    return log_policy(logits, stop, masks)

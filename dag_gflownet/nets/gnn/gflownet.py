import jax.numpy as jnp
import haiku as hk
import jraph
import math

from jax import lax, nn

from dag_gflownet.utils.gflownet import log_policy_cliques


def clique_policy(graphs, masks, k):
    """
    Parameters
    ----------
    graphs: namedtuple `Graph` of (jraph._src.graph.GraphsTuple, jraph._src.graph.GraphsTuple)
        Each element of the tuple is a batch of graphs encoded as a single GraphsTuple. 
        Each graph in the batch corresponds to the current 
        state in a parallel instantiation of the environment (there are 
        `batch_size` parallel environments).
        The distinction between the two elements in the tuple is that the first 
        element encodes in the node features the node identities, while the 
        second element encodes the node values (which are discrete).
    masks: np.ndarray of shape (batch_size, max_nodes)
        Batch of masks to prevent a given node from being sampled twice. 
        In addition, we also mask the nodes which are not part of a current incomplete 
        clique. masks[i, j] = 0 iff node j from batch i is unavailable 
        for sampling at this step. max_nodes is the maximal number of nodes to sample from. 
        In our current setting, it corresponds to the number of nodes in the ground truth graph.
    k: int
        Number of different discrete values that the nodes can take.
        
    Returns
    -------
    log_policy_cliques: jnp.DeviceArray of shape (batch_size, num_actions) = 
    (batch_size, max_nodes + 1)
        Log probabilities for each possible action, including the stop action
    """
    batch_size, max_nodes = masks.shape

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(max_nodes + k, embed_dim=128)
    edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),
        init=hk.initializers.TruncatedNormal())
    
    structural_embeddings = node_embeddings_list(graphs.structure.nodes)
    value_embeddings = node_embeddings_list(graphs.values.nodes)
    node_embeddings = structural_embeddings + value_embeddings

    graphs = graphs.structure._replace(
        nodes=node_embeddings,
        edges=jnp.repeat(edge_embedding, graphs.structure.edges.shape[0], axis=0),
        globals=jnp.zeros((graphs.structure.n_node.shape[0], 1)),
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

    # node_features = features.nodes[:batch_size * num_variables]
    global_features = features.globals[:batch_size]

    # Reshape the node features, and project into keys, queries & values
    # node_features = node_features.reshape(batch_size, num_variables, -1)
    # node_features = hk.Linear(128 * 3, name='projection')(node_features)
    # queries, keys, values = jnp.split(node_features, 3, axis=2)

    # Self-attention layer
    # node_features = hk.MultiHeadAttention(
    #     num_heads=4,
    #     key_size=32,
    #     w_init_scale=2.
    # )(queries, keys, values)
    
    # node_features = node_features.reshape(batch_size, -1)
    logits = hk.nets.MLP([128, 128, 128, max_nodes], name='logits')(global_features) # Maybe 3 layers is too much. Can try different things here. Alternative: can try using node_features instead.
    stop = hk.nets.MLP([128, 1], name='stop')(global_features)

    # Initialize the temperature parameter to 1
    temperature = hk.get_parameter('temperature', (),
        init=hk.initializers.Constant(math.log(math.expm1(1))))
    temperature = nn.softplus(temperature)

    return log_policy_cliques(logits / temperature, stop, masks)

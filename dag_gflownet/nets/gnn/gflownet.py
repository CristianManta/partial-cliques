import jax.numpy as jnp
import haiku as hk
import jraph
import math

from jax import lax, nn

from dag_gflownet.utils.gflownet import log_policy_cliques


def clique_policy(graphs, masks):
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
        Batch of masks to prevent a given node from being sampled twice by the clique policy. 
        In addition, we also mask the nodes which are not part of a current incomplete 
        clique. masks[i, j] = 0 iff node j from batch i is unavailable 
        for sampling at this step. max_nodes is the maximal number of nodes to sample from. 
        In our current setting, it corresponds to the number of nodes in the ground truth graph.
        
    Returns
    -------
    log_policy_cliques: jnp.DeviceArray of shape (batch_size, num_actions) = 
    (batch_size, max_nodes + 1)
        Log probabilities for each possible action, including the stop action
    """
    
    ############ Hardcoded params for JAX ease of debugging
    k = 2 # Number of different discrete values that the nodes can take.
    if k != 2:
        raise NotImplementedError("Some assumptions are made about k = 2 in this code. Need to re-write some parts for general k.")
    #####################################
    
    batch_size, max_nodes = masks.shape

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(max_nodes + k + 2, embed_dim=128)
    """
    + 2 because we need to reserve a special embedding for: 
    1) the (target) node with the missing value (to be sampled),     
    2) the dummy index for nodes that are "padded" so that the GraphsTuple 
        has a consistent size for JAX compilation
    """
    
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
        init=hk.initializers.Constant(1))

    return log_policy_cliques(logits / temperature, stop, masks)

def value_policy(graphs, masks):
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
        Batch of masks to prevent a given node from being sampled twice by the clique policy. 
        In addition, we also mask the nodes which are not part of a current incomplete 
        clique. masks[i, j] = 0 iff node j from batch i is unavailable 
        for sampling at this step. max_nodes is the maximal number of nodes to sample from. 
        In our current setting, it corresponds to the number of nodes in the ground truth graph.
        
    Returns
    -------
    log_policy_values: jnp.DeviceArray of shape (batch_size,)
        Log probabilities of the sampled value being 1.
        
    log_flows: jnp.DeviceArray of shape (batch_size,)
        Estimated log flow passing through the current state.
    """
    
    ############ Hardcoded params for JAX ease of debugging
    k = 2 # Number of different discrete values that the nodes can take.
    if k != 2:
        raise NotImplementedError("Some assumptions are made about k = 2 in this code. Need to re-write some parts for general k.")
    #####################################
    
    batch_size, max_nodes = masks.shape
    current_sampling_feature = max_nodes + k

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(max_nodes + k + 2, embed_dim=128)
    """
    + 2 because we need to reserve a special embedding for: 
    1) the (target) node with the missing value (to be sampled),     
    2) the dummy index for nodes that are "padded" so that the GraphsTuple 
        has a consistent size for JAX compilation
    """
    
    edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),
        init=hk.initializers.TruncatedNormal())
    
    structural_embeddings = node_embeddings_list(graphs.structure.nodes)
    value_embeddings = node_embeddings_list(graphs.values.nodes)
    node_embeddings = structural_embeddings + value_embeddings

    graphs_tuple = graphs.values._replace(
        nodes=node_embeddings,
        edges=jnp.repeat(edge_embedding, graphs.values.edges.shape[0], axis=0),
        globals=jnp.zeros((graphs.values.n_node.shape[0], 1)),
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
    features = graph_net(graphs_tuple)

    node_features = features.nodes[:batch_size * max_nodes]
    global_features = features.globals[:batch_size]

    # Project the nodes features into keys, queries & values
    node_features = hk.Linear(128 * 3, name='projection')(node_features)
    queries, keys, values = jnp.split(node_features, 3, axis=1)

    # Self-attention layer
    node_features = hk.MultiHeadAttention(
        num_heads=4,
        key_size=32,
        w_init_scale=2.
    )(queries, keys, values)
    
    all_logits = hk.nets.MLP([128, 1], name='logit')(node_features) # Assumption that k = 2 here (last layer)
    all_logits = jnp.squeeze(all_logits)
    targets = graphs.values.nodes[:batch_size * max_nodes] == current_sampling_feature # target nodes to fill values
    
    targets_ix = jnp.nonzero(targets, size=batch_size)
    target_logits = all_logits[targets_ix]
    
    log_flows = hk.nets.MLP([128, 1], name='log_flows')(global_features)
    log_flows = jnp.squeeze(log_flows)

    # Initialize the temperature parameter to 1    
    temperature = hk.get_parameter('temperature', (),
        init=hk.initializers.Constant(1))

    return (nn.log_sigmoid(target_logits / temperature), log_flows)

if __name__ == "__main__":
    import jax
    import numpy as np
    from jax import random
    from dag_gflownet.utils.jraph_utils import Graph
    
    seed = 0
    key = random.PRNGKey(seed)
    
    """
    Testing the value policy
    """    
    
    # Defining a batch of 2 structure graphs
    node_features1 = np.array([0, 1, 9, 9, 9, 9])
    senders1 = np.array([0, 1])
    receivers1 = np.array([1, 0])
    edges1 = np.array([1, 1])
    n_node1 = np.array([6])
    n_edge1 = np.array([2])
    structure_graph1 = jraph.GraphsTuple(nodes=node_features1, senders=senders1, receivers=receivers1,
    edges=edges1, n_node=n_node1, n_edge=n_edge1, globals=None)

    node_features2 = np.array([9, 9, 9, 3, 4, 5])
    senders2 = np.array([3, 4, 4, 5, 5, 3])
    receivers2 = np.array([4, 3, 5, 4, 3, 5])
    edges2 = np.array([1, 1, 1, 1, 1, 1])
    n_node2 = np.array([6])
    n_edge2 = np.array([6])
    structure_graph2 = jraph.GraphsTuple(nodes=node_features2, senders=senders2, receivers=receivers2,
    edges=edges2, n_node=n_node2, n_edge=n_edge2, globals=None)

    batched_structure_graphs = jraph.batch([structure_graph1, structure_graph2])
    

    # Defining a batch of 2 value graphs
    node_features1_val = np.array([6, 8, 9, 9, 9, 9])
    senders1_val = np.array([0, 1])
    receivers1_val = np.array([1, 0])
    edges1_val = np.array([1, 1])
    n_node1_val = np.array([6])
    n_edge1_val = np.array([2])
    value_graph1 = jraph.GraphsTuple(nodes=node_features1_val, senders=senders1_val, receivers=receivers1_val,
    edges=edges1_val, n_node=n_node1_val, n_edge=n_edge1_val, globals=None)

    node_features2_val = np.array([9, 9, 9, 6, 7, 8])
    senders2_val = np.array([3, 4, 4, 5, 5, 3])
    receivers2_val = np.array([4, 3, 5, 4, 3, 5])
    edges2_val = np.array([1, 1, 1, 1, 1, 1])
    n_node2_val = np.array([6])
    n_edge2_val = np.array([6])
    value_graph2 = jraph.GraphsTuple(nodes=node_features2_val, senders=senders2_val, receivers=receivers2_val,
    edges=edges2_val, n_node=n_node2_val, n_edge=n_edge2_val, globals=None)
    
    batched_value_graphs = jraph.batch([value_graph1, value_graph2])
    
    # Concatenating structure and values into the same data structure
    graphs = Graph(structure=batched_structure_graphs, 
                   values=batched_value_graphs)
    
    # Initializing the model
    masks = jnp.ones((2, 6), dtype=int)
    value_policy = hk.without_apply_rng(hk.transform(value_policy))
    params = value_policy.init(key, graphs, masks)
    
    # Applying the model
    log_policy_values, log_flows = jax.jit(value_policy.apply)(params, graphs, masks)
    
    """
    Testing the clique policy
    """

    # Initializing the model
    clique_policy = hk.without_apply_rng(hk.transform(clique_policy))
    params = clique_policy.init(key, graphs, masks)
    
    # Applying the model
    log_policy_cliques = jax.jit(clique_policy.apply)(params, graphs, masks) 
    
    print("done")
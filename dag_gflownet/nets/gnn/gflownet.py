import jax.numpy as jnp
import haiku as hk
import jraph
import math
from functools import partial

from jax import lax, nn, jit, vmap

from dag_gflownet.utils.gflownet import log_policy_cliques


def clique_policy(graphs, masks, x_dim, K, sampling_method=1):
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
    masks: np.ndarray of shape (batch_size, h_dim+x_dim)
        Batch of masks to prevent a given node from being sampled twice by the clique policy.
        In addition, we also mask the nodes which are not part of a current incomplete
        clique. masks[i, j] = 0 iff node j from batch i is unavailable
        for sampling at this step. h_dim is the maximal number of nodes to sample from.
        It corresponds to the number of latent nodes in the ground truth graph in our setting.
    x_dim: int
        Number of low-level variables.
    K: int
        Number of different discrete values that the nodes can take.
    sampling_method: int
        3 possible values:
        1: "policy" (default): follow the network parameters to predict the next
        node to sample
        2: "sequential": Automatically choose the first available node according to
        the mask for sampling, ignoring the learned policy.
        3: "uniform": Choose the node to sample according to an uniform distribution
        among the eligible nodes (according to the mask), ignoring the learned policy.
    Returns
    -------
    log_policy_cliques: jnp.DeviceArray of shape (batch_size, num_actions) =
    (batch_size, h_dim + 1)
        Log probabilities for each possible action, including the stop action
    """

    assert K == 2

    batch_size, num_variables = masks.shape
    h_dim = num_variables - x_dim
    masks = masks[:, :h_dim]

    if (
        sampling_method == 2
    ):  # NOTE: This section has not been tested on batch_size != 1
        assert batch_size == 1
        masking_value = -1e5
        stop = jnp.full((batch_size, 1), masking_value, dtype=float)
        first_available_node_ix = jnp.where(masks, size=1)[1]
        logits = jnp.zeros((batch_size, h_dim), dtype=float)
        logits = logits.at[0, first_available_node_ix].set(-masking_value)
        return log_policy_cliques(logits, stop, masks)

    if sampling_method == 3:
        masking_value = -1e5
        stop = jnp.full((batch_size, 1), masking_value, dtype=float)
        logits = jnp.zeros((batch_size, h_dim), dtype=float)
        return log_policy_cliques(logits, stop, masks)

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(h_dim + K + 2, embed_dim=128)
    """
    + 2 because we need to reserve a special embedding for: 
    1) the (target) node with the missing value (to be sampled),     
    2) the dummy index for nodes that are "padded" so that the GraphsTuple 
        has a consistent size for JAX compilation
    """

    edge_embedding = hk.get_parameter(
        "edge_embed", shape=(1, 128), init=hk.initializers.TruncatedNormal()
    )

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
        return hk.nets.MLP([128, 128], name="node")(features)

    @jraph.concatenated_args
    def update_edge_fn(features):
        return hk.nets.MLP([128, 128], name="edge")(features)

    @jraph.concatenated_args
    def update_global_fn(features):
        return hk.nets.MLP([128, 128], name="global")(features)

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
    logits = hk.nets.MLP([128, 128, 128, h_dim], name="logits")(
        global_features
    )  # Maybe 3 layers is too much. Can try different things here. Alternative: can try using node_features instead.
    stop = hk.nets.MLP([128, 1], name="stop")(global_features)

    # Initialize the temperature parameter to 1
    temperature = hk.get_parameter("temperature", (), init=hk.initializers.Constant(1))

    return log_policy_cliques(logits / temperature, stop, masks)


def value_policy(graphs, masks, x_dim, K):
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
    masks: np.ndarray of shape (batch_size, h_dim+x_dim)
        Batch of masks to prevent a given node from being sampled twice by the clique policy.
        In addition, we also mask the nodes which are not part of a current incomplete
        clique. masks[i, j] = 0 iff node j from batch i is unavailable
        for sampling at this step. h_dim is the maximal number of nodes to sample from.
        In our current setting, it corresponds to the number of nodes in the ground truth graph.
    x_dim: int
        Number of low-level variables.
    K: int
        Number of different discrete values that the nodes can take.
    Returns
    -------
    logits_values: jnp.DeviceArray of shape (batch_size, K)
        Unnormalized log probabilities for the categorical distribution over the
        K choices of the value for the target node to be sampled
    log_flows: jnp.DeviceArray of shape (batch_size,)
        Estimated log flow passing through the current state.
    """

    assert K == 2

    batch_size, num_variables = masks.shape
    h_dim = num_variables - x_dim
    masks = masks[:, :h_dim]
    current_sampling_feature = num_variables + K + 1

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(num_variables + K + 2, embed_dim=128)
    """
    + 2 because we need to reserve a special embedding for: 
    1) the (target) node with the missing value (to be sampled),     
    2) the dummy index for nodes that are "padded" so that the GraphsTuple 
        has a consistent size for JAX compilation
    """

    edge_embedding = hk.get_parameter(
        "edge_embed", shape=(1, 128), init=hk.initializers.TruncatedNormal()
    )

    # Preparing a separate copy of the embeddings for the flow estimator
    # Setting the target node feature to be the same as the unobserved ones
    flow_estimator_values_nodes = jnp.where(
        graphs.values.nodes == current_sampling_feature,
        num_variables + K,
        graphs.values.nodes,
    )
    flow_estimator_value_embeddings = node_embeddings_list(flow_estimator_values_nodes)

    # Embeddings for the policy head
    structural_embeddings = node_embeddings_list(graphs.structure.nodes)
    value_embeddings = node_embeddings_list(graphs.values.nodes)
    node_embeddings = structural_embeddings + value_embeddings

    # embeddings for the flow estimator
    flow_estimator_node_embeddings = (
        structural_embeddings + flow_estimator_value_embeddings
    )

    graphs_tuple = graphs.values._replace(
        nodes=node_embeddings,
        edges=jnp.repeat(edge_embedding, graphs.values.edges.shape[0], axis=0),
        globals=jnp.zeros((graphs.values.n_node.shape[0], 1)),
    )

    graphs_tuple_flow_estimator = graphs.values._replace(
        nodes=flow_estimator_node_embeddings,
        edges=jnp.repeat(edge_embedding, graphs.values.edges.shape[0], axis=0),
        globals=jnp.zeros((graphs.values.n_node.shape[0], 1)),
    )

    # Define graph network updates
    @jraph.concatenated_args
    def update_node_fn(features):
        return hk.nets.MLP([128, 128], name="node")(features)

    @jraph.concatenated_args
    def update_edge_fn(features):
        return hk.nets.MLP([128, 128], name="edge")(features)

    @jraph.concatenated_args
    def update_global_fn(features):
        return hk.nets.MLP([128, 128], name="global")(features)

    graph_net = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
    )
    features = graph_net(graphs_tuple)
    features_flow_estimator = graph_net(graphs_tuple_flow_estimator)

    node_features = features.nodes[: batch_size * num_variables]
    global_features = features_flow_estimator.globals[
        :batch_size
    ]  # We don't want the flow estimator to know about the node targeted for sampling its value

    # Project the nodes features into keys, queries & values
    node_features = hk.Linear(128 * 3, name="projection")(node_features)
    queries, keys, values = jnp.split(node_features, 3, axis=1)

    # Self-attention layer
    node_features = hk.MultiHeadAttention(num_heads=4, key_size=32, w_init_scale=2.0)(
        queries, keys, values
    )

    all_logits = hk.nets.MLP([128, K], name="logit")(
        node_features
    )  # Assumption that k = 2 here (last layer)
    # all_logits = jnp.squeeze(all_logits)
    targets = (
        graphs.values.nodes[: batch_size * num_variables] == current_sampling_feature
    )  # target nodes to fill values

    targets_ix = jnp.nonzero(targets, size=batch_size)
    target_logits = all_logits[targets_ix]

    log_flows = hk.nets.MLP([128, 1], name="log_flows")(global_features)
    log_flows = jnp.squeeze(log_flows, axis=1)

    # Initialize the temperature parameter to 1
    temperature = hk.get_parameter("temperature", (), init=hk.initializers.Constant(1))

    return (target_logits / temperature, log_flows)


def value_policy_MLP(graphs, masks, x_dim, K):
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
    masks: np.ndarray of shape (batch_size, h_dim+x_dim)
        Batch of masks to prevent a given node from being sampled twice by the clique policy.
        In addition, we also mask the nodes which are not part of a current incomplete
        clique. masks[i, j] = 0 iff node j from batch i is unavailable
        for sampling at this step. h_dim is the maximal number of nodes to sample from.
        In our current setting, it corresponds to the number of nodes in the ground truth graph.
    x_dim: int
        Number of low-level variables.
    K: int
        Number of different discrete values that the nodes can take.
    Returns
    -------
    logits_values: jnp.DeviceArray of shape (batch_size, K)
        Unnormalized log probabilities for the categorical distribution over the
        K choices of the value for the target node to be sampled
    log_flows: jnp.DeviceArray of shape (batch_size,)
        Estimated log flow passing through the current state.
    """

    ############# Parameters

    embedding_dimensions = 256

    #######################

    assert K == 2

    batch_size, num_variables = masks.shape
    h_dim = num_variables - x_dim
    masks = masks[:, :h_dim]
    current_sampling_feature = num_variables + K + 1

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(
        num_variables + K + 2, embed_dim=embedding_dimensions
    )
    """
    + 2 because we need to reserve a special embedding for: 
    1) the (target) node with the missing value (to be sampled),     
    2) the dummy index for nodes that are "padded" so that the GraphsTuple 
        has a consistent size for JAX compilation
    """

    # Preparing a separate copy of the embeddings for the flow estimator
    # Setting the target node feature to be the same as the unobserved ones
    flow_estimator_values_nodes = jnp.where(
        graphs.values.nodes == current_sampling_feature,
        num_variables + K,
        graphs.values.nodes,
    )
    flow_estimator_value_embeddings = node_embeddings_list(flow_estimator_values_nodes)

    # Embeddings for the policy head
    structural_embeddings = node_embeddings_list(graphs.structure.nodes)
    value_embeddings = node_embeddings_list(graphs.values.nodes)
    node_embeddings = structural_embeddings + value_embeddings

    # embeddings for the flow estimator
    flow_estimator_node_embeddings = (
        structural_embeddings + flow_estimator_value_embeddings
    )

    split_empeddings = jnp.array(jnp.split(node_embeddings, batch_size))
    x_h_merged_embeddings = jnp.reshape(split_empeddings, (batch_size, -1))

    split_empeddings_flow = jnp.array(
        jnp.split(flow_estimator_node_embeddings, batch_size)
    )
    x_h_merged_embeddings_flow = jnp.reshape(split_empeddings_flow, (batch_size, -1))

    node_features = hk.nets.MLP(
        [embedding_dimensions, embedding_dimensions], name="node_features"
    )(x_h_merged_embeddings)
    global_features = hk.nets.MLP(
        [embedding_dimensions, embedding_dimensions], name="global_features"
    )(x_h_merged_embeddings_flow)

    all_logits = hk.nets.MLP([embedding_dimensions, K], name="logit")(node_features)

    # Initialize the temperature parameter to 1
    temperature = hk.get_parameter("temperature", (), init=hk.initializers.Constant(1))

    log_flows = hk.nets.MLP([embedding_dimensions, 1], name="log_flows")(
        global_features
    )
    log_flows = jnp.squeeze(log_flows, axis=1)

    return (all_logits / temperature, log_flows)

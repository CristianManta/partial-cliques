import jax.numpy as jnp
import haiku as hk

from dag_gflownet.nets.transformer.transformers import Transformer


def value_policy_transformer(graphs, masks, x_dim, K):
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

    transformer = Transformer(num_heads=4, num_layers=6, key_size=32, dropout_rate=0.0)

    # Embedding of the nodes & edges
    node_embeddings_list = hk.Embed(num_variables + K + 2, embed_dim=128)
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
    node_embeddings = jnp.reshape(
        structural_embeddings + value_embeddings, (batch_size, -1, 128)
    )

    # embeddings for the flow estimator
    flow_estimator_node_embeddings = jnp.reshape(
        structural_embeddings + flow_estimator_value_embeddings, (batch_size, -1, 128)
    )

    node_features = transformer(node_embeddings)
    flow_estimator_node_features = transformer(flow_estimator_node_embeddings)

    all_logits = hk.nets.MLP([128, K], name="logit")(node_features)
    all_logits = jnp.reshape(
        all_logits, (batch_size * num_variables, -1)
    )  # Prepare for indexing with targets_ix

    targets = (
        graphs.values.nodes[: batch_size * num_variables] == current_sampling_feature
    )

    targets_ix = jnp.nonzero(targets, size=batch_size)

    # OUTPUT: Computing policy logits
    target_logits = all_logits[targets_ix]

    # Initialize the temperature parameter to 1
    temperature = hk.get_parameter("temperature", (), init=hk.initializers.Constant(1))

    # OUTPUT: Computing log_flows
    flow_estimator_node_features = jnp.reshape(
        flow_estimator_node_features, (batch_size, num_variables * 128)
    )

    log_flows = hk.nets.MLP([num_variables * 128, 1], name="log_flows")(
        flow_estimator_node_features
    )
    log_flows = jnp.squeeze(log_flows, axis=1)

    return (target_logits / temperature, log_flows)

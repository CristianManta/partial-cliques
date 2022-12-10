import pytest
import numpy as np

from dag_gflownet.utils.data import (
    get_clique_selection_mask,
    get_value_policy_energy,
    get_random_graph,
    get_potential_fns,
)


def test_get_clique_selection_mask():
    seed = 0
    np.random.seed(seed)

    K = 2
    # Setting up a dummy GFN state
    # Assuming that we have 10 variables, x_0^3 and h_0^5
    # We have two cliques {x_0^3, h_0^2} and {x_0^3, h_3^5}
    # We have only fully observed x_0^3 and h_0
    gfn_state = (
        np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 2, 2, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]

    mask = get_clique_selection_mask(gfn_state, unobserved_cliques, K)
    assert np.all(mask == np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])), mask


def test_get_value_policy_energy():
    seed = 0
    np.random.seed(seed)

    K = 2
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]

    # Setting up a list of dummy potential functions
    clique_potentials = [lambda c: 0.2 if len(c) == 7 else np.nan, lambda c: -2]
    gfn_state = (
        np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    unobserved_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]
    full_cliques = [set([0, 1, 2, 6, 7, 8, 9]), set([3, 4, 5, 6, 7, 8, 9])]
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, unobserved_cliques, full_cliques, clique_potentials, K
    )
    assert energy == -np.log(0.2)
    assert len(new_unobserved_cliques[0]) == 0
    gfn_state = (
        np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 0, 2, 0, 1, 0, 1]),
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
    )
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, new_unobserved_cliques, full_cliques, clique_potentials, K
    )
    assert energy == 0.0

    # Test potential fns
    model, (full_cliques, _), data = get_random_graph(d=6, D=4, n=4)
    unobserved_cliques = full_cliques.copy()
    gfn_state = (
        np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 1, 0, 2, 2, 2, 0, 1, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )

    clique_potentials = get_potential_fns(model, unobserved_cliques)
    new_gfn_state, new_unobserved_cliques, energy = get_value_policy_energy(
        gfn_state, unobserved_cliques, full_cliques, clique_potentials, K
    )

    target_energy = 0
    for c_ind, c in enumerate(new_unobserved_cliques):
        if len(c) == 0:
            target_energy += clique_potentials[c_ind].values[
                (tuple(gfn_state[1][np.array(sorted(list(full_cliques[c_ind])))]))
            ]
    assert energy == target_energy

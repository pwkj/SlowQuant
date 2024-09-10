from collections.abc import Sequence
import numpy as np
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    G4,
    G5,
    G6)
from slowquant.variational_coupled_cluster.util import VccStructure
import scipy.sparse as ss
from slowquant.unitary_coupled_cluster.operator_matrix import get_indexing, build_operator_matrix
import functools

def construct_vcc_state(
    state: np.ndarray,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    vcc_struct: VccStructure,
    dagger: bool = False,
) -> np.ndarray:
    """Contruct transformation matrix.

    Args:
        num_det: Number of determinants.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).

    Returns:
        Unitary transformation matrix.
    """
    T = np.zeros((len(state), len(state)))
    for exc_type, exc_indices, theta in zip(
        vcc_struct.excitation_operator_type, vcc_struct.excitation_indicies, thetas
    ):
        if abs(theta) < 10**-14:
            continue
        if exc_type == "single":
            (i, a) = exc_indices
            T += (
                theta
                * G1_matrix(i, a, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
            )
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T += (
                theta
                * G2_matrix(
                    i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            )
        elif exc_type == "triple":
            (i, j, k, a, b, c) = exc_indices
            T += (
                theta
                * G3_matrix(
                    i, j, k, a, b, c, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            )
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = exc_indices
            T += (
                theta
                * G4_matrix(
                    i,
                    j,
                    k,
                    l,
                    a,
                    b,
                    c,
                    d,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = exc_indices
            T += (
                theta
                * G5_matrix(
                    i,
                    j,
                    k,
                    l,
                    m,
                    a,
                    b,
                    c,
                    d,
                    e,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = exc_indices
            T += (
                theta
                * G6_matrix(
                    i,
                    j,
                    k,
                    l,
                    m,
                    n,
                    a,
                    b,
                    c,
                    d,
                    e,
                    f,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    if dagger:
        return ss.linalg.expm_multiply(-T, state)
    return ss.linalg.expm_multiply(T, state)


@functools.cache
def G1_matrix(i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T1 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G1(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)


@functools.cache
def G2_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)


@functools.cache
def G3_matrix(
    i: int,
    j: int,
    k: int,
    a: int,
    b: int,
    c: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T3 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G3(i, j, k, a, b, c), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)


@functools.cache
def G4_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    a: int,
    b: int,
    c: int,
    d: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T4 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G4(i, j, k, l, a, b, c, d), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)


@functools.cache
def G5_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T5 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G5(i, j, k, l, m, a, b, c, d, e), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)


@functools.cache
def G6_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    n: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T6 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        n: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        f: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G6(i, j, k, l, m, n, a, b, c, d, e, f), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op)

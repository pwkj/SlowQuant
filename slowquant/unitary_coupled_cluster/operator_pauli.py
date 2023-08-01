from __future__ import annotations

import copy
import functools

import numpy as np
import scipy.sparse as ss

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.base import (
    StateVector,
    kronecker_product_cached,
    pauli_to_mat,
)


@functools.cache
def a_spin_pauli(idx: int, dagger: bool, num_spin_orbs: int, num_elec: int) -> OperatorPauli:
    """Annihilation operator with spin orbital index.

    Args:
        idx: Spin orbital index.
        dagger: If complex conjugated.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        Pauli operator.
    """
    if idx % 2 == 0:
        return a_pauli(idx // 2, 'alpha', dagger, num_spin_orbs, num_elec)
    else:
        return a_pauli((idx - 1) // 2, 'beta', dagger, num_spin_orbs, num_elec)


@functools.cache
def a_pauli(spinless_idx: int, spin: str, dagger: bool, num_spin_orbs: int, num_elec: int) -> OperatorPauli:
    """Annihilation operator.

    Args:
        spinless_idx: Spatial orbital index.
        spin: alpha or beta spin.
        dagger: If complex conjugated.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        Pauli operator.
    """
    idx = 2 * spinless_idx
    if spin == 'beta':
        idx += 1
    operators = {}
    op1 = ''
    op2 = ''
    fac1: complex = 1
    fac2: complex = 1
    for i in range(num_spin_orbs):
        if i == idx:
            if dagger:
                op1 += 'X'
                fac1 *= 0.5
                op2 += 'Y'
                fac2 *= -0.5j
            else:
                op1 += 'X'
                fac1 *= 0.5
                op2 += 'Y'
                fac2 *= 0.5j
        elif i < idx:
            op1 += 'Z'
            op2 += 'Z'
        else:
            op1 += 'I'
            op2 += 'I'
    operators[op1] = fac1
    operators[op2] = fac2
    return OperatorPauli(operators)


def expectation_value_pauli(
    bra: StateVector,
    pauliop: OperatorPauli,
    ket: StateVector,
    use_csr: int = 10,
) -> float:
    """Calculate expectation value of Pauli operator.

    Args:
       bra: Bra state-vector.
       pauliop: Pauli operator.
       ket: Ket state-vector.
       use_csr: Size for when to use sparse matrices.

    Returns:
        Expectation value of Pauli operator.
    """
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError('Bra and Ket does not have same number of inactive orbitals')
    if len(bra._active) != len(ket._active):
        raise ValueError('Bra and Ket does not have same number of active orbitals')
    total: complex = 0.0
    for op, fac in pauliop.operators.items():
        if abs(fac) < 10**-12:
            continue
        tmp = 1.0
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(pauli_to_mat(op[i]), ket.ket_inactive[:, i]))  # type: ignore
        for i in range(len(bra.bra_virtual)):
            op_idx = i + len(bra.bra_inactive) + len(bra._active_onvector)
            tmp *= np.matmul(bra.bra_virtual[i], np.matmul(pauli_to_mat(op[op_idx]), ket.ket_virtual[:, i]))  # type: ignore
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        active_start = len(bra.bra_inactive)
        active_end = active_start + number_active_orbitals
        tmp_active = 1.0
        active_pauli_string = op[active_start:active_end]
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.deepcopy(ket.ket_active_csr)
            else:
                operator = copy.deepcopy(ket.ket_active)
            for pauli_mat_idx, pauli_mat_symbol in enumerate(active_pauli_string):
                prior = pauli_mat_idx
                after = number_active_orbitals - pauli_mat_idx - 1
                if pauli_mat_symbol == 'I':
                    continue
                if number_active_orbitals >= use_csr:
                    operator = kronecker_product_cached(prior, after, pauli_mat_symbol, True).dot(operator)
                else:
                    operator = np.matmul(
                        kronecker_product_cached(prior, after, pauli_mat_symbol, False),
                        operator,
                    )
            if number_active_orbitals >= use_csr:
                tmp_active *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                tmp_active *= np.matmul(bra.bra_active, operator)
        total += fac * tmp * tmp_active
    if abs(total.imag) > 10**-10:
        print(f'WARNING: Imaginary value of {total.imag}')
    return total.real


class OperatorPauli:
    def __init__(self, operator: dict[str, complex]) -> None:
        """Initialize Pauli operator.

        Args:
            operator: Pauli operator in dictionary form.
        """
        self.operators = operator
        self.screen_zero = True

    def __add__(self, pauliop: OperatorPauli) -> OperatorPauli:
        """Overload addition operator.

        Args:
            pauliop: Pauli opertor.

        Returns:
            New Pauli Operator
        """
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] += fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = fac
        return OperatorPauli(new_operators)

    def __sub__(self, pauliop: OperatorPauli) -> OperatorPauli:
        """Overload subtraction operator.

        Args:
            pauliop: Pauli opertor.

        Returns:
            New Pauli Operator
        """
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] -= fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = -fac
        return OperatorPauli(new_operators)

    def __mul__(self, pauliop: OperatorPauli) -> OperatorPauli:
        """Overload multiplication operator.

        Args:
            pauliop: Pauli opertor.

        Returns:
            New Pauli Operator
        """
        new_operators: dict[str, complex] = {}
        for op1, val1 in self.operators.items():
            for op2, val2 in pauliop.operators.items():
                new_op = ''
                fac: complex = val1 * val2
                for pauli1, pauli2 in zip(op1, op2):
                    if pauli1 == 'I':
                        new_op += pauli2
                    elif pauli2 == 'I':
                        new_op += pauli1
                    elif pauli1 == pauli2:
                        new_op += 'I'
                    elif pauli1 == 'X' and pauli2 == 'Y':
                        new_op += 'Z'
                        fac *= 1j
                    elif pauli1 == 'X' and pauli2 == 'Z':
                        new_op += 'Y'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'X':
                        new_op += 'Z'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'Z':
                        new_op += 'X'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'X':
                        new_op += 'Y'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'Y':
                        new_op += 'X'
                        fac *= -1j
                if new_op in new_operators:
                    new_operators[new_op] += fac
                else:
                    new_operators[new_op] = fac
                if self.screen_zero:
                    if abs(new_operators[new_op]) < 10**-12:
                        del new_operators[new_op]
        return OperatorPauli(new_operators)

    def __rmul__(self, number: float) -> OperatorPauli:
        """Overload right multiplication operator.

        Args:
            number: Scalar value.

        Returns:
            New Pauli Operator
        """
        new_operators = self.operators.copy()
        for op in self.operators:
            new_operators[op] *= number
            if self.screen_zero:
                if abs(new_operators[op]) < 10**-12:
                    del new_operators[op]
        return OperatorPauli(new_operators)

    @property
    def dagger(self) -> OperatorPauli:
        """Do complex conjugate of Pauli operator.

        Returns:
            New Pauli operator.
        """
        new_operators = {}
        for op, fac in self.operators.items():
            new_operators[op] = np.conj(fac)
        return OperatorPauli(new_operators)

    def eval_operators(self, state_vector: StateVector) -> dict[str, float]:
        """Evalate operator per term.

        Args:
            state_vector: Bra and Ket state-vector.

        Returns:
            Expectation value of all terms in Pauli operator.
        """
        op_values = {}
        for op in self.operators:
            op_values[op] = expectation_value_pauli(state_vector, OperatorPauli({op: 1}), state_vector)
        return op_values

    def matrix_form(self, use_csr: int = 10, is_real: bool = False) -> np.ndarray | ss.csr_matrix:
        """Get matrix form of operator.

        Args:
            use_csr: Size for when to use sparse matrices.
            is_real: Only return real part of matrix.

        Returns:
            Pauli operator in matrix form.
        """
        num_spin_orbs = len(list(self.operators.keys())[0])
        if num_spin_orbs >= use_csr:
            matrix_form = ss.identity(2**num_spin_orbs, dtype=complex) * 0.0
        else:
            matrix_form = np.identity(2**num_spin_orbs, dtype=complex) * 0.0
        for op, fac in self.operators.items():
            if abs(fac) < 10**-12:
                continue
            if num_spin_orbs >= use_csr:
                tmp = ss.identity(2**num_spin_orbs, dtype=complex)
            else:
                tmp = np.identity(2**num_spin_orbs, dtype=complex)
            for pauli_mat_idx, pauli_mat_symbol in enumerate(op):
                prior = pauli_mat_idx
                after = num_spin_orbs - pauli_mat_idx - 1
                if pauli_mat_symbol == 'I':
                    continue
                if num_spin_orbs >= use_csr:
                    A = kronecker_product_cached(prior, after, pauli_mat_symbol, True)
                    tmp = A.dot(tmp)
                else:
                    tmp = np.matmul(kronecker_product_cached(prior, after, pauli_mat_symbol, False), tmp)
            matrix_form += fac * tmp
        if num_spin_orbs >= use_csr:
            if matrix_form.getformat() != 'csr':
                matrix_form = ss.csr_matrix(matrix_form)
        if is_real:
            matrix_form = matrix_form.astype(float)
        return matrix_form

    def screen_terms(
        self, max_xy_inactive: int, max_xy_virtual: int, num_inactive_spin_orbs: int, num_virtual_spin_orbs
    ) -> OperatorPauli:
        """Remove terms from the operator that has too many X or Y Pauli operators.

        If it is known that a maximum number of Pauli operators can be changed before
        the expectation value of the operator is taken.
        Then terms with too many X and Y operators can be removed because they will always evaluate to zero.

        Args:
            max_xy_inactive: Maximum number of X and Y Pauli operators in the inactive orbitals.
            max_xy_virtual: Maximum number of X and Y Pauli operators in the virtual orbitals.
            num_inactive_spin_orbs: Number of inactive orbitals.
            num_virtual_spin_orbs: Number of virtual orbitals.

        Returns:
            Screened Pauli operator.
        """
        new_operators = {}
        for op, fac in self.operators.items():
            if (
                op[:num_inactive_spin_orbs].count('X') + op[:num_inactive_spin_orbs].count('Y')
                > max_xy_inactive
            ):
                continue
            # Only check if number of virtual are more than zero.
            # Indexing with -0 returns the entire string; 'XYZ'[-0:] = 'XYZ'
            if num_virtual_spin_orbs > 0:
                if (
                    op[-num_virtual_spin_orbs:].count('X') + op[-num_virtual_spin_orbs:].count('Y')
                    > max_xy_virtual
                ):
                    continue
            new_operators[op] = fac
        return OperatorPauli(new_operators)


def epq_pauli(p: int, q: int, num_spin_orbs: int, num_elec: int) -> OperatorPauli:
    """Get Epq operator.

    Args:
        p: Orbital index.
        q: Orbital index.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        Epq Pauli operator.
    """
    E = a_pauli(p, 'alpha', True, num_spin_orbs, num_elec) * a_pauli(
        q, 'alpha', False, num_spin_orbs, num_elec
    )
    E += a_pauli(p, 'beta', True, num_spin_orbs, num_elec) * a_pauli(
        q, 'beta', False, num_spin_orbs, num_elec
    )
    return E


def epqrs_pauli(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int) -> OperatorPauli:
    """Get epqrs operator.

    Args:
        p: Orbital index.
        q: Orbital index.
        r: Orbital index.
        s: Orbital index.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        epqrs Pauli operator.
    """
    if p == r and q == s:
        operator = 2 * (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
        )
    elif p == q == r:
        operator = (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(s, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', False, num_spin_orbs, num_elec)
        )
        operator += (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(p, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(s, 'beta', False, num_spin_orbs, num_elec)
        )
    elif p == r == s:
        operator = (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(p, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
        )
        operator += (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', False, num_spin_orbs, num_elec)
        )
    elif q == s:
        operator = (
            a_pauli(r, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
        )
        operator += (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(r, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
        )
    else:
        operator = (
            a_pauli(r, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(s, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
        )
        operator += (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(r, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(s, 'beta', False, num_spin_orbs, num_elec)
        )
        operator -= (
            a_pauli(p, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(r, 'beta', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'beta', False, num_spin_orbs, num_elec)
            * a_pauli(s, 'beta', False, num_spin_orbs, num_elec)
        )
        operator -= (
            a_pauli(p, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(r, 'alpha', True, num_spin_orbs, num_elec)
            * a_pauli(q, 'alpha', False, num_spin_orbs, num_elec)
            * a_pauli(s, 'alpha', False, num_spin_orbs, num_elec)
        )
    return operator


def hamiltonian_pauli(
    h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int
) -> OperatorPauli:
    """Get full Hamiltonian operator.

    Args:
        h: One-electron Hamiltonian integrals in AO.
        g: Two-electron Hamiltonian integrals in AO.
        c_mo: Orbital coefficients.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        Full Hamilonian Pauli operator.
    """
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_spatial_orbs = num_spin_orbs // 2
    H_expectation = OperatorPauli({})
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            if abs(h_mo[p, q]) > 10**-12:
                H_expectation += h_mo[p, q] * epq_pauli(p, q, num_spin_orbs, num_elec)
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            for r in range(num_spatial_orbs):
                for s in range(num_spatial_orbs):
                    if abs(g_mo[p, q, r, s]) > 10**-12:
                        H_expectation += (
                            1 / 2 * g_mo[p, q, r, s] * epqrs_pauli(p, q, r, s, num_spin_orbs, num_elec)
                        )
    return H_expectation


def commutator_pauli(A: OperatorPauli, B: OperatorPauli) -> OperatorPauli:
    """Calculate commutator of two Pauli operators.

    Args:
        A: Pauli operator.
        B: Pauli operator.

    Returns:
        New Pauli operator.
    """
    return A * B - B * A


def energy_hamiltonian_pauli(
    h: np.ndarray,
    g: np.ndarray,
    c_mo: np.ndarray,
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
) -> OperatorPauli:
    """Get energy Hamiltonian operator.

    Args:
        h: One-electron Hamiltonian integrals in AO.
        g: Two-electron Hamiltonian integrals in AO.
        c_mo: Orbital coefficients.
        num_inactive_spin_orbs: Number of inactive spin orbitals.
        num_active_spin_orbs: Number active spin orbitals.
        num_virtual_spin_orbs: Number of virtual spin orbitals.
        num_elec: Number of electrons.

    Returns:
        Energy Hamilonian Pauli operator.
    """
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_inactive_spatial_orbs = num_inactive_spin_orbs // 2
    num_active_spatial_orbs = num_active_spin_orbs // 2
    num_spin_orbs = num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs
    H_expectation = OperatorPauli({})
    # Inactive one-electron
    for i in range(num_inactive_spatial_orbs):
        if abs(h_mo[i, i]) > 10**-12:
            H_expectation += h_mo[i, i] * epq_pauli(i, i, num_spin_orbs, num_elec)
    # Active one-electron
    for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
        for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            if abs(h_mo[p, q]) > 10**-12:
                H_expectation += h_mo[p, q] * epq_pauli(p, q, num_spin_orbs, num_elec)
    # Inactive two-electron
    for i in range(num_inactive_spatial_orbs):
        for j in range(num_inactive_spatial_orbs):
            if abs(g_mo[i, i, j, j]) > 10**-12:
                H_expectation += 1 / 2 * g_mo[i, i, j, j] * epqrs_pauli(i, i, j, j, num_spin_orbs, num_elec)
            if i != j and abs(g_mo[j, i, i, j]) > 10**-12:
                H_expectation += 1 / 2 * g_mo[j, i, i, j] * epqrs_pauli(j, i, i, j, num_spin_orbs, num_elec)
    # Inactive-Active two-electron
    for i in range(num_inactive_spatial_orbs):
        for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
                if abs(g_mo[i, i, p, q]) > 10**-12:
                    H_expectation += (
                        1 / 2 * g_mo[i, i, p, q] * epqrs_pauli(i, i, p, q, num_spin_orbs, num_elec)
                    )
                if abs(g_mo[p, q, i, i]) > 10**-12:
                    H_expectation += (
                        1 / 2 * g_mo[p, q, i, i] * epqrs_pauli(p, q, i, i, num_spin_orbs, num_elec)
                    )
                if abs(g_mo[p, i, i, q]) > 10**-12:
                    H_expectation += (
                        1 / 2 * g_mo[p, i, i, q] * epqrs_pauli(p, i, i, q, num_spin_orbs, num_elec)
                    )
                if abs(g_mo[i, p, q, i]) > 10**-12:
                    H_expectation += (
                        1 / 2 * g_mo[i, p, q, i] * epqrs_pauli(i, p, q, i, num_spin_orbs, num_elec)
                    )
    # Active two-electron
    for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
        for q in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
            for r in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
                for s in range(
                    num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs
                ):
                    if abs(g_mo[p, q, r, s]) > 10**-12:
                        H_expectation += (
                            1 / 2 * g_mo[p, q, r, s] * epqrs_pauli(p, q, r, s, num_spin_orbs, num_elec)
                        )
    return H_expectation.screen_terms(0, 0, num_inactive_spin_orbs, num_virtual_spin_orbs)
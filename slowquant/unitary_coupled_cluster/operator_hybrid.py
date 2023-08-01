from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import StateVector, pauli_to_mat
from slowquant.unitary_coupled_cluster.operator_pauli import OperatorPauli


def expectation_value_hybrid(
    bra: StateVector, hybridop: OperatorHybrid, ket: StateVector, use_csr: int = 10
) -> float:
    """Calculate expectation value of hybrid operator.

    Args:
        bra: Bra state-vector.
        hybridop: Hybrid operator.
        ket: Ket state-vector.
        use_csr: Size when to use sparse matrices.

    Returns:
        Expectation value of hybrid operator.
    """
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError('Bra and Ket does not have same number of inactive orbitals')
    if len(bra._active) != len(ket._active):
        raise ValueError('Bra and Ket does not have same number of active orbitals')
    total = 0
    for _, op in hybridop.operators.items():
        tmp = 1
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(
                bra.bra_inactive[i], np.matmul(pauli_to_mat(op.inactive_pauli[i]), ket.ket_inactive[:, i])  # type: ignore
            )
        for i in range(len(bra.bra_virtual)):
            tmp *= np.matmul(
                bra.bra_virtual[i], np.matmul(pauli_to_mat(op.virtual_pauli[i]), ket.ket_virtual[:, i])  # type: ignore
            )
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.deepcopy(ket.ket_active_csr)
                operator = op.active_matrix.dot(operator)
                tmp *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                operator = copy.deepcopy(ket.ket_active)
                operator = np.matmul(op.active_matrix, operator)
                tmp *= np.matmul(bra.bra_active, operator)
        total += tmp
    if abs(total.imag) > 10**-10:
        print(f'WARNING, imaginary value of {total.imag}')
    return total.real


def convert_pauli_to_hybrid_form(
    pauliop: OperatorPauli, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
) -> OperatorHybrid:
    """Convert Pauli operator to hybrid operator.

    Args:
        pauliop: Pauli operator.
        num_inactive_orbs: Number of inactive orbitals.
        num_active_orbs: Number of active orbitals.
        num_virtual_orbs: Number of virtual orbitals.

    Returns:
        Hybrid operator.
    """
    new_operator: dict[str, np.ndarray] = {}
    active_start = num_inactive_orbs
    active_end = num_inactive_orbs + num_active_orbs
    for pauli_string, factor in pauliop.operators.items():
        new_inactive = pauli_string[:active_start]
        new_active = pauli_string[active_start:active_end]
        new_virtual = pauli_string[active_end:]
        active_pauli = OperatorPauli({new_active: 1})
        new_active_matrix = factor * active_pauli.matrix_form()
        key = new_inactive + new_virtual
        if key in new_operator:
            new_operator[key].active_matrix += new_active_matrix
        else:
            new_operator[key] = OperatorHybridData(new_inactive, new_active_matrix, new_virtual)
    return OperatorHybrid(new_operator)


class OperatorHybridData:
    def __init__(
        self, inactive_pauli: str, active_matrix: np.ndarray | ss.csr_matrix, virtual_pauli: str
    ) -> None:
        """Initialize data structure of hybrid operators.

        Args:
            inactive_pauli: Pauli string of inactive orbitals.
            active_matrix: Matrix operator of active orbitals.
            virtual_pauli: Pauli string of virtual orbitals.
        """
        self.inactive_pauli = inactive_pauli
        self.active_matrix = active_matrix
        self.virtual_pauli = virtual_pauli


class OperatorHybrid:
    def __init__(self, operator: dict[str, OperatorHybridData]) -> None:
        """Initialize hybrid operator.

        The key is the Pauli-string of inactive + virtual,
        i.e. the active part does not contribute to the key.

        Args:
            operator: Dictonary form of hybrid operator.
        """
        self.operators = operator

    def __add__(self, hybridop: OperatorHybrid) -> OperatorHybrid:
        """Overload addition operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix += op.active_matrix
            else:
                new_operators[key] = OperatorHybridData(op.inactive_pauli, op.active_matrix, op.virtual_pauli)
        return OperatorHybrid(new_operators)

    def __sub__(self, hybridop: OperatorHybrid) -> OperatorHybrid:
        """Overload subtraction operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix -= op.active_matrix
            else:
                new_operators[key] = OperatorHybridData(
                    op.inactive_pauli, -op.active_matrix, op.virtual_pauli
                )
        return OperatorHybrid(new_operators)

    def __mul__(self, pauliop: OperatorHybrid) -> OperatorHybrid:
        """Overload multiplication operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators: dict[str, np.ndarray] = {}
        for _, op1 in self.operators.items():
            for _, op2 in pauliop.operators.items():
                new_inactive = ''
                new_virtual = ''
                fac: complex = 1
                for pauli1, pauli2 in zip(op1.inactive_pauli, op2.inactive_pauli):
                    if pauli1 == 'I':
                        new_inactive += pauli2
                    elif pauli2 == 'I':
                        new_inactive += pauli1
                    elif pauli1 == pauli2:
                        new_inactive += 'I'
                    elif pauli1 == 'X' and pauli2 == 'Y':
                        new_inactive += 'Z'
                        fac *= 1j
                    elif pauli1 == 'X' and pauli2 == 'Z':
                        new_inactive += 'Y'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'X':
                        new_inactive += 'Z'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'Z':
                        new_inactive += 'X'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'X':
                        new_inactive += 'Y'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'Y':
                        new_inactive += 'X'
                        fac *= -1j
                for pauli1, pauli2 in zip(op1.virtual_pauli, op2.virtual_pauli):
                    if pauli1 == 'I':
                        new_virtual += pauli2
                    elif pauli2 == 'I':
                        new_virtual += pauli1
                    elif pauli1 == pauli2:
                        new_virtual += 'I'
                    elif pauli1 == 'X' and pauli2 == 'Y':
                        new_virtual += 'Z'
                        fac *= 1j
                    elif pauli1 == 'X' and pauli2 == 'Z':
                        new_virtual += 'Y'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'X':
                        new_virtual += 'Z'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'Z':
                        new_virtual += 'X'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'X':
                        new_virtual += 'Y'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'Y':
                        new_virtual += 'X'
                        fac *= -1j
                new_active = fac * lw.matmul(op1.active_matrix, op2.active_matrix)
                key = new_inactive + new_virtual
                if key in new_operators:
                    new_operators[key].active_matrix += new_active
                else:
                    new_operators[key] = OperatorHybridData(new_inactive, new_active, new_virtual)
        return OperatorHybrid(new_operators)

    def __rmul__(self, number: float) -> OperatorHybrid:
        """Overload right multiplication operator.

        Args:
            number: Scalar value.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix *= number
        return OperatorHybrid(new_operators)

    @property
    def dagger(self) -> OperatorHybrid:
        """Do complex conjugate of operator.

        Returns:
            New hybrid operator.
        """
        new_operators = {}
        for key, op in self.operators.items():
            new_operators[key] = OperatorHybridData(
                op.inactive_pauli, np.conj(op.active_matrix).transpose(), op.virtual_pauli
            )
        return OperatorHybrid(new_operators)

    def apply_u_from_right(self, U: np.ndarray | ss.csr_matrix) -> OperatorHybrid:
        """Matrix multiply with transformation matrix from the right.

        Args:
            U: Transformation matrix.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = lw.matmul(new_operators[key].active_matrix, U)
        return OperatorHybrid(new_operators)

    def apply_u_from_left(self, U: np.ndarray | ss.csr_matrix) -> OperatorHybrid:
        """Matrix multiply with transformation matrix from the left.

        Args:
            U: Transformation matrix.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = lw.matmul(U, new_operators[key].active_matrix)
        return OperatorHybrid(new_operators)
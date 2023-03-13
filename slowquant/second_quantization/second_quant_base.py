from __future__ import annotations

import copy

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)


class a_op:
    def __init__(self, spinless_idx: int, spin: str, dagger: bool) -> None:
        """Initialize fermionic annihilation operator.

        Args:
            spinless_idx: Spatial orbital index.
            spin: Alpha or beta spin.
            dagger: If creation operator.
        """
        self.spinless_idx = spinless_idx
        self.idx = 2 * self.spinless_idx
        self.dagger = dagger
        self.spin = spin
        if self.spin == "beta":
            self.idx += 1


def operator_string_to_key(operator_string: list[a_op]) -> str:
    """Make key string to index a fermionic operator in a dict structure.

    Args:
        operator_string: Fermionic opreators.

    Returns:
        Dictionary key.
    """
    string_key = ""
    for a in operator_string:
        if a.dagger:
            string_key += f"c{a.idx}"
        else:
            string_key += f"a{a.idx}"
    return string_key


def check_always_zero(operator: list[a_op]) -> bool:
    max_idx = 0
    for a in operator:
        max_idx = max(max_idx, a.idx + 1)
    state = [0] * max_idx
    for a in operator[::-1]:
        idx = a.idx
        if a.dagger:
            state[idx] += 1
            if state[idx] == 2:
                return True
        else:
            state[idx] -= 1
            if state[idx] == -2:
                return True
    return False


class FermionicOperator:
    def __init__(self, annihilation_operator: dict[str, list[a_op]], factor: dict[str, float]) -> None:
        if not isinstance(annihilation_operator, dict):
            string_key = operator_string_to_key([annihilation_operator])
            self.operators = {}
            self.operators[string_key] = [annihilation_operator]
            self.factors = {}
            self.factors[string_key] = factor
        else:
            self.operators = annihilation_operator
            self.factors = factor

    def __add__(self, fermistring: FermionicOperator) -> FermionicOperator:
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for string_key in fermistring.operators.keys():
            if string_key in operators.keys():
                factors[string_key] += fermistring.factors[string_key]
                if abs(factors[string_key]) < 10**-14:
                    del factors[string_key]
                    del operators[string_key]
            else:
                operators[string_key] = fermistring.operators[string_key]
                factors[string_key] = fermistring.factors[string_key]
        return FermionicOperator(operators, factors)

    def __sub__(self, fermistring: FermionicOperator) -> FermionicOperator:
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for string_key in fermistring.operators.keys():
            if string_key in operators.keys():
                factors[string_key] -= fermistring.factors[string_key]
                if abs(factors[string_key]) < 10**-14:
                    del factors[string_key]
                    del operators[string_key]
            else:
                operators[string_key] = fermistring.operators[string_key]
                factors[string_key] = -fermistring.factors[string_key]
        return FermionicOperator(operators, factors)

    def __mul__(self, fermistring: FermionicOperator) -> FermionicOperator:
        operators = {}
        factors = {}
        for string_key1 in fermistring.operators.keys():
            for string_key2 in self.operators.keys():
                str_key = string_key2 + string_key1
                if str_key not in operators.keys():
                    operators[str_key] = self.operators[string_key2] + fermistring.operators[string_key1]
                    factors[str_key] = self.factors[string_key2] * fermistring.factors[string_key1]
                    if check_always_zero(operators[str_key]):
                        del operators[str_key]
                        del factors[str_key]
                else:
                    factors[str_key] += self.factors[string_key2] * fermistring.factors[string_key1]
                    if abs(factors[str_key]) < 10**-14:
                        del factors[str_key]
                        del operators[str_key]
        return FermionicOperator(operators, factors)

    def __rmul__(self, number: float) -> FermionicOperator:
        operators = {}
        factors = {}
        for key_string in self.operators:
            operators[key_string] = self.operators[key_string]
            factors[key_string] = self.factors[key_string] * number
        return FermionicOperator(operators, factors)

    def get_operator_count(self) -> dict[int, int]:
        op_count = {}
        for string_key in self.operators.keys():
            op_lenght = len(self.operators[string_key])
            if op_lenght not in op_count:
                op_count[op_lenght] = 1
            else:
                op_count[op_lenght] += 1
        return op_count


class WaveFunction:
    def __init__(self, number_spin_orbitals: int) -> None:
        self.determinants = []
        self.coefficients = []
        self.number_spin_orbitals = number_spin_orbitals
        self.number_spatial_orbitals = number_spin_orbitals // 2
        self.kappa = np.zeros((self.number_spatial_orbitals, self.number_spatial_orbitals))
        self.c_mo = scipy.linalg.expm(-self.kappa)

    def add_determinant(self, determinant: list[int], coefficient: float) -> None:
        self.determinants.append(determinant)
        self.coefficients.append(coefficient)

    def get_non_redundant_kappa(self) -> list[int, int]:
        kappa_index = []
        for p in range(self.number_spatial_orbitals):
            for q in range(self.number_spatial_orbitals):
                if p <= q:
                    continue
                if len(apply_on_ket(Eminuspq(p, q), self)[0]) == 0:
                    continue
                kappa_index.append([p, q])
        return kappa_index

    def update_kappa(self, kappa_values: list[float], kappa_indicies: list[int, int]) -> None:
        for value, (p, q) in zip(kappa_values, kappa_indicies):
            if p == q:
                print("WARNING: kappa_pp is changed")
            self.kappa[p, q] = value
            self.kappa[q, p] = -value
        self.c_mo = np.matmul(self.c_mo, scipy.linalg.expm(-self.kappa))
        self.kappa[:, :] = 0.0


def collapse_operator_on_determinant(operator: list[a_op], determinant: list[int]) -> tuple[list[int], int]:
    determinant_out = np.array(determinant).astype(int)
    phase = 1
    for a in operator[::-1]:
        idx = a.idx
        if a.dagger:
            if determinant_out[idx] == 1:
                return [], 0
            determinant_out[idx] = 1
        else:
            if determinant_out[idx] == 0:
                return [], 0
            determinant_out[idx] = 0
        if np.sum(determinant_out[:idx]) % 2 == 1:
            phase *= -1
    return determinant_out.tolist(), phase


def apply_on_ket(operators: FermionicOperator, ket: WaveFunction) -> tuple[list[int], list[int]]:
    determinants = []
    phases = []
    for key_string in operators.operators.keys():
        for ket_determinant in ket.determinants:
            collapsed_determinant, phase = collapse_operator_on_determinant(
                operators.operators[key_string], ket_determinant
            )
            if phase == 0:
                continue
            determinants.append(collapsed_determinant)
            phases.append(phase)
    return determinants, phases


def Epq(p: int, q: int) -> FermionicOperator:
    E = FermionicOperator(a_op(p, "alpha", dagger=True), 1) * FermionicOperator(
        a_op(q, "alpha", dagger=False), 1
    )
    E += FermionicOperator(a_op(p, "beta", dagger=True), 1) * FermionicOperator(
        a_op(q, "beta", dagger=False), 1
    )
    return E


def epqrs(p: int, q: int, r: int, s: int) -> FermionicOperator:
    if q == r:
        return Epq(p, q) * Epq(r, s) - Epq(p, s)
    return Epq(p, q) * Epq(r, s)


def Eminuspq(p: int, q: int) -> FermionicOperator:
    return Epq(p, q) - Epq(q, p)


def H(h: np.ndarray, g: np.ndarray, c_mo: np.ndarray) -> FermionicOperator:
    H_operator = FermionicOperator({}, {})
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_bf = len(c_mo)
    for p in range(num_bf):
        for q in range(num_bf):
            H_operator += h_mo[p, q] * Epq(p, q)
    for p in range(num_bf):
        for q in range(num_bf):
            for r in range(num_bf):
                for s in range(num_bf):
                    H_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return H_operator


def comm(A: FermionicOperator, B: FermionicOperator) -> FermionicOperator:
    return A * B - B * A

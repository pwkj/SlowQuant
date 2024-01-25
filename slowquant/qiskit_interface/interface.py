import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.circuit.library import PUCCD, UCC, UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.custom_ansatz import (
    ErikD_JW,
    ErikD_Parity,
    ErikSD_JW,
    ErikSD_Parity,
)


class QuantumInterface:
    """Quantum interface class.

    This class handles the interface with qiskit and the communication with quantum hardware.
    """

    def __init__(
        self,
        primitive: BaseEstimator | BaseSampler,
        ansatz: str,
        mapper: FermionicMapper,
        do_shot_balancing: bool = False,
        precision: float = 1e-3,
        confidence: float = 0.68,
    ) -> None:
        """
        Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            primitive: Qiskit Estimator or Sampler object
            ansatz: Name of qiskit ansatz to be used. Currenly supported: UCCSD, UCCD, and PUCCD
            mapper: Qiskit mapper object, e.g. JW or Parity
            do_shot_balancing: Use shot balancing instead of a uniform number of shots when measuring the Pauli strings.
            precision: Fermionic expectation value precision to target with shot balancing.
            confidence: Confidence that wanted precision is reached with shot balancing.
        """
        allowed_ansatz = ("UCCSD", "PUCCD", "UCCD", "ErikD", "ErikSD", "HF")
        if ansatz not in allowed_ansatz:
            raise ValueError("The chosen Ansatz is not availbale. Choose from: ", allowed_ansatz)
        self.ansatz = ansatz
        self.primitive = primitive
        self.mapper = mapper
        self.precision = precision
        self.confidence = confidence
        self.do_shot_balancing = do_shot_balancing

    def construct_circuit(self, num_orbs: int, num_parts: int) -> None:
        """
        Construct qiskit circuit

        Args:
            num_orbs: number of orbitals
            num_parts: number of particles/electrons
        """
        self.num_orbs = (
            num_orbs  # that might be a dirty and stupid solution for the num_orbs problem. revisit it!
        )

        if self.ansatz == "UCCSD":
            self.circuit = UCCSD(
                num_orbs,
                (num_parts // 2, num_parts // 2),
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    (num_parts // 2, num_parts // 2),
                    self.mapper,
                ),
            )
        elif self.ansatz == "PUCCD":
            self.circuit = PUCCD(
                num_orbs,
                (num_parts // 2, num_parts // 2),
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    (num_parts // 2, num_parts // 2),
                    self.mapper,
                ),
            )
        elif self.ansatz == "UCCD":
            self.circuit = UCC(
                num_orbs,
                (num_parts // 2, num_parts // 2),
                "d",
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    (num_parts // 2, num_parts // 2),
                    self.mapper,
                ),
            )
        elif self.ansatz == "ErikD":
            if num_orbs != 2 or num_parts != 2:
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "ErikSD":
            if num_orbs != 2 or num_parts != 2:
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikSD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikSD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "HF":
            self.circuit = HartreeFock(num_orbs, (num_parts // 2, num_parts // 2), self.mapper)

        # Set parameter to HarteeFock
        self._parameters = [0.0] * self.circuit.num_parameters

    @property
    def parameters(self) -> list[float]:
        """Get ansatz parameters.

        Returns:
            Ansatz parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(
        self,
        parameters: list[float],
    ) -> None:
        """Set ansatz parameters.

        Args:
            parameters: List of ansatz parameters.
        """
        if len(parameters) != self.circuit.num_parameters:
            raise ValueError(
                "The length of the parameter list does not fit the chosen circuit for the Ansatz ",
                self.ansatz,
            )
        self._parameters = parameters.copy()

    def op_to_qbit(self, op: FermionicOperator) -> SparsePauliOp:
        """
        Fermionic operator to qbit rep

        Args:
            op: Operator as SlowQuant's FermionicOperator object
        """
        return self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs), 2 * self.num_orbs))

    def quantum_expectation_value(
        self, op: FermionicOperator, custom_parameters: list[float] | None = None
    ) -> float:
        """Calculate expectation value of circuit and observables.

        Args:
            op: Operator as SlowQuant's FermionicOperator object.

        Returns:
            Expectation value of fermionic operator.
        """
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters

        # Check if estimator or sampler
        if isinstance(self.primitive, BaseEstimator):
            return self._estimator_quantum_expectation_value(op, run_parameters)
        elif isinstance(self.primitive, BaseSampler):
            if self.do_shot_balancing:
                return self._sampler_quantum_expectation_value_balanced(op, run_parameters)
            else:
                return self._sampler_quantum_expectation_value(op, run_parameters)
        else:
            raise ValueError(
                "The Quantum Interface was initiated with an unknown Qiskit primitive, {type(self.primitive)}"
            )

    def _estimator_quantum_expectation_value(
        self, op: FermionicOperator, run_parameters: list[float]
    ) -> float:
        """Calculate expectation value of circuit and observables via Estimator.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        job = self.primitive.run(
            circuits=self.circuit,
            parameter_values=run_parameters,
            observables=self.op_to_qbit(op),
        )
        result = job.result()
        values = result.values[0]

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value(self, op: FermionicOperator, run_parameters: list[float]) -> float:
        """Calculate expectation value of circuit and observables via Sampler.

        The expectation value over a fermionic operator is calcuated as:

        .. math::
            E = \\sum_i^N c_i\\left<0\\left|P_i\\right|0\\right>

        With :math:`c_i` being the :math:`i` the coefficient and :math:`P_i` the :math:`i` the Pauli string.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        values = 0.0
        observables = self.op_to_qbit(op)

        # Loop over all qubit-mapped Paul strings and get Sampler distributions
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            values += self._sampler_distributions(pauli, run_parameters) * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value_balanced(
        self, op: FermionicOperator, run_parameters: list[float]
    ) -> float:
        """Calculate expectation value of circuit and observables via Sampler using shot balancing.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        values = 0.0
        observables = self.op_to_qbit(op)
        # The -2 is because only I and only Z operators have in principle always zero variance.
        n_p = len(observables.paulis) - 2
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            p1_new = self._sampler_distribution_p1(pauli, run_parameters, 1000)
            p1 = p1_new
            n_tot = 1000
            sigma_p = 2 * np.abs(coeff) * (p1 - p1**2) ** (1 / 2)
            n = self.confidence**2 * n_p * sigma_p**2 / self.precision**2
            n_shots = int(max(n / 2 - n_tot, 0))
            if n_shots != 0:
                p1_new = self._sampler_distribution_p1(pauli, run_parameters, n_shots)
                p1 = (n_tot * p1 + p1_new * n_shots) / (n_tot + n_shots)
                n_tot += n_shots
                sigma_p = 2 * np.abs(coeff) * (p1 - p1**2) ** (1 / 2)
            while self.confidence * (n_p) ** (1 / 2) * sigma_p / (n_tot) ** (1 / 2) > self.precision:
                n = max(self.confidence**2 * n_p * sigma_p**2 / self.precision**2, 1000)
                n_shots = int(max(1.1 * n - n_tot, 0.1 * n))
                p1_new = self._sampler_distribution_p1(pauli, run_parameters, n_shots)
                p1 = (n_tot * p1 + p1_new * n_shots) / (n_tot + n_shots)
                n_tot += n_shots
                sigma_p = 2 * np.abs(coeff) * (p1 - p1**2) ** (1 / 2)
            values += 2 * coeff * p1 - coeff
            print(pauli, n_tot, p1)
        return values.real

    def _sampler_distribution_p1(self, pauli: Pauli, run_parameters: list[float], shots: int) -> float:
        """Sample the probability of measuring one for a given Pauli string.

        Args:
            pauli: Pauli string.
            run_paramters: Ansatz parameters.
            shots: Number of shots.

        Returns:
            p1 probability.
        """
        # Create QuantumCircuit
        ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
        ansatz_w_obs.measure_all()

        # Run sampler
        self.primitive.set_options(shots=shots)
        job = self.primitive.run(ansatz_w_obs, parameter_values=run_parameters)

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0].binary_probabilities()

        p1 = 0.0
        for key, value in distr.items():
            # Here we could check if we want a given key (bitstring) in the result distribution
            if get_bitstring_sign(pauli, key) == 1:
                p1 += value
        # should prob. also return actual number of shots used, if this information is returned from device
        return p1

    def _sampler_distributions(self, pauli: Pauli, run_parameters: list[float]) -> float:
        """Get results from a sampler distribution for one given Pauli string.

        The expectation value of a Pauli string is calcuated as:

        .. math::
            E = \\sum_i^N p_i\\left<b_i\\left|P\\right|b_i\\right>

        With :math:`p_i` being the :math:`i` th probability and :math:`b_i` being the `i` th bit-string.

        Args:
            pauli: Pauli string to measure.
            run_paramters: Parameters of circuit.

        Returns:
            Probability weighted Pauli string.
        """
        # Create QuantumCircuit
        ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
        ansatz_w_obs.measure_all()

        # Run sampler
        job = self.primitive.run(ansatz_w_obs, parameter_values=run_parameters)

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0].binary_probabilities()

        result = 0.0
        for key, value in distr.items():
            # Here we could check if we want a given key (bitstring) in the result distribution
            result += value * get_bitstring_sign(pauli, key)
        return result


def to_CBS_measurement(op: Pauli) -> QuantumCircuit:
    """Convert a Pauli string to Pauli measurement circuit.

    This is achived by the following transformation:

    .. math::
        \\begin{align}
        I &\\rightarrow I\\\\
        Z &\\rightarrow Z\\\\
        X &\\rightarrow XH\\\\
        Y &\\rightarrow YS^{\\dagger}H
        \\end{align}

    Args:
        op: Pauli string operator.

    Returns:
        Pauli measuremnt quantum circuit.
    """
    num_qubits = len(op)
    qc = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(op):
        if pauli == Pauli("X"):
            qc.append(pauli, [i])
            qc.h(i)
        elif pauli == Pauli("Y"):
            qc.append(pauli, [i])
            qc.sdg(i)
            qc.h(i)
    return qc


def get_bitstring_sign(op: Pauli, binary: str) -> int:
    """Convert Pauli string and bit-string measurement to expectation value.

    Takes Pauli String and a state in binary form and returns the sign based on the expectation value of the Pauli string with each single quibit state.

    This is achived by using the following evaluations:

    .. math::
        \\begin{align}
        \\left<0\\left|I\\right|0\\right> &= 1\\\\
        \\left<1\\left|I\\right|1\\right> &= 1\\\\
        \\left<0\\left|Z\\right|0\\right> &= 1\\\\
        \\left<1\\left|Z\\right|1\\right> &= -1\\\\
        \\left<0\\left|HXH\\right|0\\right> &= 1\\\\
        \\left<1\\left|HXH\\right|1\\right> &= -1\\\\
        \\left<0\\left|HSYS^{\\dagger}H\\right|0\\right> &= 1\\\\
        \\left<1\\left|HSYS^{\\dagger}H\\right|1\\right> &= -1
        \\end{align}

    The total expectation value is then evaulated as:

    .. math::
        E = \\prod_i^N\\left<b_i\\left|P_{i,T}\\right|b_i\\right>

    With :math:`b_i` being the :math:`i` th bit and :math:`P_{i,T}` being the :math:`i` th proberly transformed Pauli operator.

    Args:
        op: Pauli string operator.
        binary: Measured bit-string.

    Returns:
        Expectation value of Pauli string.
    """
    sign = 1
    for i, pauli in enumerate(op.to_label()):
        if not pauli == "I":
            if binary[i] == "1":
                sign = sign * (-1)
    return sign

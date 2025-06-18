import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UccStructure,
    UpsStructure,
    iterate_t1_sa,
    iterate_t2_sa,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
)


class SubspaceExpansion:
    index_info: tuple[CI_Info, list[float], UpsStructure] | tuple[CI_Info, list[float], UccStructure]

    def __init__(
        self, wave_function: WaveFunctionUCC | WaveFunctionUPS, excitations: str, do_TDA: bool = False
    ) -> None:
        """Initialize subspace expansion by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            do_TDA: Apply Tamm-Dancoff approximation (default: False).
        """
        self.wf = wave_function
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")
        self.G_ops: list[FermionicOperator] = [] #[FermionicOperator({"": []}, {"": 1.0})]
        self.G_ops_d: list[FermionicOperator] = [] #[FermionicOperator({"": []}, {"": 1.0})]
        excitations = excitations.lower()
        if "s" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa(i, a))
                if not do_TDA:
                    self.G_ops_d.append(G1_sa(i, a).dagger)
                print(i, a)
                #print(G1_sa(i, a).get_qiskit_form(num_orbs=4))
               
        if "d" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                if op_type == 1:
                    self.G_ops.append(G2_1_sa(i, j, a, b))
                    if not do_TDA:
                        self.G_ops_d.append(G2_1_sa(i, j, a, b).dagger)
                elif op_type == 2:
                    self.G_ops.append(G2_2_sa(i, j, a, b))
                    if not do_TDA:
                        self.G_ops_d.append(G2_2_sa(i, j, a, b).dagger)
                print(a, i, b, j)
                #print(G2_1_sa(i, j, a, b).get_qiskit_form(num_orbs=4))
        
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
                if not do_TDA:
                    self.G_ops.append(G3(i, j, k, a, b, c).dagger)
        if "q" in excitations:
            for a, i, b, j, c, k, d, l in iterate_t4(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G4(i, j, k, l, a, b, c, d))
                if not do_TDA:
                    self.G_ops.append(G4(i, j, k, l, a, b, c, d).dagger)
        if "5" in excitations:
            for a, i, b, j, c, k, d, l, e, m in iterate_t5(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e))
                if not do_TDA:
                    self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e).dagger)
        if "6" in excitations:
            for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f))
                if not do_TDA:
                    self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f).dagger)
        num_parameters = len(self.G_ops)

        H = np.zeros((num_parameters+1, num_parameters+1))
        S = np.zeros((num_parameters+1, num_parameters+1))
     
        if not do_TDA:
            H = np.zeros((2*num_parameters+1, 2*num_parameters+1))
            S = np.zeros((2*num_parameters+1, 2*num_parameters+1))

        H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )

        G_ket = []
        Gd_ket = []
        HG_ket = []
        HGd_ket = []
        for i in range(num_parameters):
            GJ_ket = propagate_state([self.G_ops[i]], self.wf.ci_coeffs, *self.index_info)
            G_ket.append(GJ_ket)
            HG_ket.append(propagate_state([H_0i_0a], GJ_ket, *self.index_info))
            if not do_TDA:
                GJd_ket = propagate_state([self.G_ops_d[i]], self.wf.ci_coeffs, *self.index_info)
                Gd_ket.append(GJd_ket)
                HGd_ket.append(propagate_state([H_0i_0a], GJd_ket, *self.index_info))
            
        #H[0,0]=E_0 and S[0,0]=1:
        ket = propagate_state([], self.wf.ci_coeffs, *self.index_info)
        H[0, 0] = expectation_value(ket, [H_0i_0a], ket,*self.index_info,)
        S[0, 0] = expectation_value(ket,[],ket,*self.index_info,)

        #H[0,j>0] and S[0,j>0]:
        for j, GI_ket in enumerate(G_ket):
            # Make H
            #<Gd_jH>
            H[0, j+1] = expectation_value(HG_ket[j],[],ket,*self.index_info,)
            H[j+1, 0] = H[0, j+1].conjugate()
            if not do_TDA:
                #<G_jH>
                H[0, j+1+num_parameters] = expectation_value(HGd_ket[j],[],ket,*self.index_info,)
                H[j+1+num_parameters, 0] = H[0, j+1+num_parameters].conjugate()
            # Make S
            #<Gd_j>
            S[0, j+1] = expectation_value(GI_ket,[],ket,*self.index_info,)
            S[j+1, 0] = S[0, j+1].conjugate()
            if not do_TDA:
                #<G_j>
                S[0, j+1+num_parameters] = expectation_value(ket,[],GI_ket,*self.index_info,)
                S[j+1+num_parameters, 0] = S[0, j+1+num_parameters].conjugate()

        #H[i>0,j>0]: 
        for i, GI_ket in enumerate(G_ket):
             for j, HGJ_ket in enumerate(HG_ket):
                if j>=i:
                    # Make H
                    #<Gd_iHG_j>
                    H[i+1, j+1] = expectation_value(GI_ket,[],HGJ_ket,*self.index_info,)
                    H[j+1, i+1] = H[i+1, j+1].conjugate()
                    if not do_TDA:
                        #<G_iHG_j>
                        H[i+1, j+1+num_parameters] = expectation_value(Gd_ket[i],[],HGJ_ket,*self.index_info,)
                        H[j+1+num_parameters, i+1] = H[i+1, j+1+num_parameters].conjugate()
                        #<G_iHGd_j>
                        H[i+1+num_parameters, j+1+num_parameters] = expectation_value(Gd_ket[i],[],HGd_ket[j],*self.index_info,)
                        H[j+1+num_parameters, i+1+num_parameters] = H[i+1+num_parameters, j+1+num_parameters].conjugate()

        #S[i>0,j>0]:
        for i, GI_ket in enumerate(G_ket):
             for j, GJ_ket in enumerate(G_ket):
                if j>=i:
                    # Make S
                    #<Gd_iG_j>
                    S[i+1, j+1] = expectation_value(GI_ket,[],GJ_ket,*self.index_info,)
                    S[j+1, i+1] = S[i+1, j+1].conjugate()
                    print(i+1, j+1)
                    if not do_TDA:
                        print(i+1, j+1+num_parameters)
                        #<G_iG_j>
                        S[i+1, j+1+num_parameters] = expectation_value(Gd_ket[i],[],GJ_ket,*self.index_info,)
                        S[j+1+num_parameters, i+1] = S[i+1, j+1+num_parameters].conjugate()
                        #<G_iGd_j>
                        S[i+1+num_parameters, j+1+num_parameters] = expectation_value(Gd_ket[i],[],Gd_ket[j],*self.index_info,)
                        S[j+1+num_parameters, i+1+num_parameters] = S[i+1+num_parameters, j+1+num_parameters].conjugate()


        self.H = H
        self.S = S
        eigval, _ = scipy.linalg.eig(H, S)
        sorting = np.argsort(eigval)
        eigval = eigval[sorting]
        self.E0 = np.real(eigval[0])
        self.excitation_energies = np.real(eigval[1:]) - self.E0




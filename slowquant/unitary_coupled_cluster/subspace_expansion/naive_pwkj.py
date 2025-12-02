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
    Epq,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
    hamiltonian_0i_0a,
    hamiltonian_1i_1a,
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
        self, wave_function: WaveFunctionUCC | WaveFunctionUPS, excitations: str
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
        
        self.G_ops: list[FermionicOperator] = []
        self.q_ops: list[FermionicOperator] = []
        excitations = excitations.lower()

        if "s" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa(i, a))
        if "d" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                if op_type == 1:
                    self.G_ops.append(G2_1_sa(i, j, a, b))
                elif op_type == 2:
                    self.G_ops.append(G2_2_sa(i, j, a, b))
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
        if "q" in excitations:
            for a, i, b, j, c, k, d, l in iterate_t4(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G4(i, j, k, l, a, b, c, d))
        if "5" in excitations:
            for a, i, b, j, c, k, d, l, e, m in iterate_t5(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e))
        if "6" in excitations:
            for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f))
        for i, a in self.wf.kappa_no_activeactive_idx:
            op = 2 ** (-1 / 2) * Epq(a, i)
            self.q_ops.append(op)
            #print(op.get_qiskit_form(num_orbs=6)) 

        q_ops_num_parameters = len(self.q_ops)
        G_ops_num_parameters = len(self.G_ops)
       

        H = np.zeros((2*G_ops_num_parameters+q_ops_num_parameters+1, 2*G_ops_num_parameters+q_ops_num_parameters+1))
        S = np.zeros((2*G_ops_num_parameters+q_ops_num_parameters+1, 2*G_ops_num_parameters+q_ops_num_parameters+1))
        

        H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )

        H_1i_1a = hamiltonian_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs
        )


        G_ket = []
        Gd_ket = []
        HG_ket = []
        GdH_ket = []
        HGd_ket = []
        for j, op in enumerate(self.G_ops):
            G_ket.append(propagate_state([op], self.wf.ci_coeffs, *self.index_info))
            Gd_ket.append(propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info))
            HG_ket.append(propagate_state([H_0i_0a * op], self.wf.ci_coeffs, *self.index_info))
            GdH_ket.append(propagate_state([op.dagger * H_0i_0a], self.wf.ci_coeffs, *self.index_info))
            HGd_ket.append(propagate_state([H_0i_0a * op.dagger], self.wf.ci_coeffs, *self.index_info))

        q_ket = []
        Hq_ket = []
        qdH_ket = []
        for j, qJ in enumerate(self.q_ops):
            #print(self.q_ops[i].get_qiskit_form(num_orbs=6))
            q_ket.append(propagate_state([qJ], self.wf.ci_coeffs, *self.index_info))
            Hq_ket.append(propagate_state([H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info))
            qdH_ket.append(propagate_state([qJ.dagger * H_1i_1a], self.wf.ci_coeffs, *self.index_info))
           

        #H[0,0]=E_0 and S[0,0]=1:
        ket = propagate_state([], self.wf.ci_coeffs, *self.index_info)
        H[0, 0] = expectation_value(ket, [H_0i_0a], ket,*self.index_info,)
        S[0, 0] = expectation_value(ket,[],ket,*self.index_info,)

        # Make H
        for j in range(q_ops_num_parameters):
            #<0|Hq|0>
            H[0, j+1] = expectation_value(ket,[],Hq_ket[j],*self.index_info,)
            #print(i+1)
            for i in range(q_ops_num_parameters):
                #<0|qdHq|0>
                H[i+1, j+1] = expectation_value(q_ket[i],[],Hq_ket[j],*self.index_info,)
                #print(i+1, j+1)

        for i in range(q_ops_num_parameters): 
            for j in range(G_ops_num_parameters):
                #<0|qdHG|0>
                H[i+1, q_ops_num_parameters+j+1] = expectation_value(q_ket[i],[],HG_ket[j],*self.index_info,)
                #print(q_ops_num_parameters+i+1, j+1)
                #<0|qdHGd|0>
                H[i+1, G_ops_num_parameters+q_ops_num_parameters+j+1] = expectation_value(q_ket[i],[],GdH_ket[j],*self.index_info,)

        for j in range(G_ops_num_parameters):
            # <0|HG|0>
            H[0, q_ops_num_parameters+j+1] = expectation_value(ket,[],HG_ket[j],*self.index_info,)
            # <0|HGd|0>
            H[0, G_ops_num_parameters+q_ops_num_parameters+j+1] = expectation_value(ket,[],GdH_ket[j],*self.index_info,)
            for i in range(G_ops_num_parameters):
                # <0|GdHG|0>
                H[q_ops_num_parameters+i+1, q_ops_num_parameters+j+1] = expectation_value(G_ket[i],[],HG_ket[j],*self.index_info,)
                # <0|GdHGd|0>
                H[q_ops_num_parameters+i+1, G_ops_num_parameters+q_ops_num_parameters+j+1] = expectation_value(HG_ket[i],[],Gd_ket[j],*self.index_info,)
                # <0|GHGd|0>
                H[G_ops_num_parameters+q_ops_num_parameters+i+1, G_ops_num_parameters+q_ops_num_parameters+j+1] = expectation_value(HGd_ket[i],[],Gd_ket[j],*self.index_info,)

            

        #     #<Gd_jH> and <G_jH>
        #     H[0, i+1] = expectation_value(HG_ket[i],[],ket,*self.index_info,)
        #     H[i+1, 0] = H[0, i+1].conjugate()
        #     #<G_jH> and <HGd_j>
        #     H[0, i+1+num_parameters] = expectation_value(HGd_ket[i],[],ket,*self.index_info,)  
        #     H[i+1+num_parameters,0] = H[0, i+1+num_parameters]
        #     # Make S
        #     #<Gd_j> and <G_j> 
        #     S[0, i+1] = expectation_value(G_ket[i],[],ket,*self.index_info,)
        #     S[i+1, 0] = S[0, i+1].conjugate()
        #     #<G_j> and <Gd_j>
        #     S[0, i+1+num_parameters] = expectation_value(ket,[],G_ket[i],*self.index_info,)
        #     S[i+1+num_parameters, 0] = S[0, i+1+num_parameters].conjugate()
            
        # #H[i>0,j>0] and S[i>0,j>0]: 
        # if not do_TDA:
        #     for i in range(num_parameters):
        #         for j in range(num_parameters):
        #             if j>=i:
        #                 # Make H
        #                 # <Gd_iHG_j> 
        #                 H[i+1, j+1] = expectation_value(G_ket[j],[],HG_ket[i],*self.index_info,)
        #                 H[j+1, i+1] = H[i+1, j+1].conjugate()
        #                 # Make S
        #                 # <Gd_iG_j>
        #                 S[i+1, j+1] = expectation_value(G_ket[j],[],G_ket[i],*self.index_info,)
        #                 S[j+1, i+1] = S[i+1, j+1].conjugate()
        #             # Make H
        #             # <G_iHG_j> 
        #             H[i+1, j+1+num_parameters] = expectation_value(Gd_ket[j],[],HG_ket[i],*self.index_info,)
        #             # Make S
        #             # <G_iG_j>
        #             S[i+1, j+1+num_parameters] = expectation_value(Gd_ket[j],[],G_ket[i],*self.index_info,)
                   

        #     for i in range(num_parameters):
        #         for j in range(num_parameters):
        #             if j>=i:
        #                 # Make H
        #                 #<G_iHGd_j>
        #                 H[i+1+num_parameters, j+1+num_parameters] = expectation_value(Gd_ket[j],[],HGd_ket[i],*self.index_info,)
        #                 H[j+1+num_parameters, i+1+num_parameters] = H[i+1+num_parameters, j+1+num_parameters].conjugate()
        #                 # Make S
        #                 # <G_iGd_j>
        #                 S[i+1+num_parameters, j+1+num_parameters] = expectation_value(Gd_ket[j],[],Gd_ket[i],*self.index_info,)
        #                 S[j+1+num_parameters, i+1+num_parameters] = S[i+1+num_parameters, j+1+num_parameters].conjugate()

        #             # Make H
        #             # <Gd_iHGd_j> 
        #             H[i+1+num_parameters, j+1] = expectation_value(G_ket[j],[],HGd_ket[i],*self.index_info,)
        #             # Make S
        #             # <Gd_iGd_j>
        #             S[i+1+num_parameters, j+1] = expectation_value(G_ket[j],[],Gd_ket[i],*self.index_info,)
                    
                           
        #     self.H = H
        #     self.S = S  

        # eigval, _ = scipy.linalg.eig(H, S)
        # sorting = np.argsort(eigval)
        # eigval = eigval[sorting]
        # self.E0 = np.real(eigval[0])
        # self.excitation_energies = np.real(eigval[1:]) - self.E0







import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import (FermionicOperator, a_op)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
    hamiltonian_0i_0a,
    hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UccStructure,
    UpsStructure,
    iterate_t1,
    iterate_t1_sa_generalized,
    iterate_t1_sa,
    iterate_t2,
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
        self.G_ops: list[FermionicOperator] = [] #[FermionicOperator({"": []}, {"": 1.0})]
        I = FermionicOperator({"": []}, {"": 1.0})
        excitations = excitations.lower()
        if "s" in excitations:
            #for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
            for a, i in iterate_t1(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                #print(a, i)
                # # See Eq. (11) in ArXiv 2505.00883
                # G_a = FermionicOperator(a_op(a, "alpha", dagger=True), 1) * FermionicOperator(a_op(i, "alpha", dagger=False), 1)
                # G_a -= G_a.dagger

                # G_b = FermionicOperator(a_op(a, "beta", dagger=True), 1) * FermionicOperator(a_op(i, "beta", dagger=False), 1)
                # G_b -= G_b.dagger

                G = G1(i, a, return_anti_hermitian = True)
                self.G_ops.append(I+np.sin(1)*G-(np.cos(1)-1)*G*G)
                #print(G.get_qiskit_form(num_orbs=self.wf.num_orbs))

                # U_a = I+np.sin(1)*G_a-(np.cos(1)-1)*G_a*G_a
                # U_b = I+np.sin(1)*G_b-(np.cos(1)-1)*G_b*G_b
                # self.G_ops.append(U_a*U_b)

        if "d" in excitations:
            for a, i, b, j in iterate_t2(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                #print(a, b, i, j)
                G = G2(i, j, a, b, return_anti_hermitian = True)
                self.G_ops.append(I+np.sin(1)*G-(np.cos(1)-1)*G*G)

        num_parameters = len(self.G_ops)

        H = np.zeros((num_parameters+1, num_parameters+1))
        S = np.zeros((num_parameters+1, num_parameters+1))
        
        H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )

        #H_0i_0a = hamiltonian_full_space(self.wf.h_mo, self.wf.g_mo, self.wf.num_orbs) 

        G_ket = []
        HG_ket = []
        for i in range(num_parameters):
            GJ_ket = propagate_state([self.G_ops[i]], self.wf.ci_coeffs, *self.index_info)
            G_ket.append(GJ_ket)
            HG_ket.append(propagate_state([H_0i_0a], GJ_ket, *self.index_info))
                
        #H[0,0]=E_0 and S[0,0]=1:
        ket = propagate_state([], self.wf.ci_coeffs, *self.index_info)
        H[0, 0] = expectation_value(ket, [H_0i_0a], ket,*self.index_info,)
        S[0, 0] = expectation_value(ket,[],ket,*self.index_info,)

        #H[0,j>0] and S[0,j>0]:
        for j in range(num_parameters):
            # Make H
            #<Gd_jH>
            H[0, j+1] = expectation_value(HG_ket[j],[],ket,*self.index_info,)
            H[j+1, 0] = H[0, j+1].conjugate()
            # Make S
            #<Gd_j> 
            S[0, j+1] = expectation_value(G_ket[j],[],ket,*self.index_info,)
            S[j+1, 0] = S[0, j+1].conjugate()

        
        #H[i>0,j>0] and S[i>0,j>0]: 
        for i in range(num_parameters):
             for j in range(num_parameters):
                if j>=i:
                    # Make H
                    #<Gd_iHG_j> 
                    H[i+1, j+1] = expectation_value(G_ket[i],[],HG_ket[j],*self.index_info,)
                    H[j+1, i+1] = H[i+1, j+1].conjugate()
                    # Make S
                    #<Gd_iG_j>
                    if i==j:
                        S[i+1, j+1] = 1.0
                    else:
                        S[i+1, j+1] = expectation_value(G_ket[i],[],G_ket[j],*self.index_info,)
                        S[j+1, i+1] = S[i+1, j+1].conjugate()
      
                   
        self.H = H
        self.S = S  

        eigval, _ = scipy.linalg.eig(H, S)
        sorting = np.argsort(eigval)
        eigval = eigval[sorting]
        self.E0 = np.real(eigval[0])
        self.excitation_energies = np.real(eigval[1:]) - self.E0







from collections.abc import Sequence
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2, iterate_t3, iterate_t4, iterate_t5, iterate_t6, iterate_pair_t2
import copy

class VccStructure:
    def __init__(self) -> None:
        """Intialize the unitary coupled cluster ansatz structure."""
        self.excitation_indicies: list[tuple[int, ...]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params = 0

    def reorder(self, num_spin_orbs):
        new_excitation_indicies = []
        new_excitation_operator_type = []

        tmp1_idx = copy.deepcopy(self.excitation_indicies)
        tmp1_type = copy.deepcopy(self.excitation_operator_type)
        for i in range(num_spin_orbs-1, -1, -1):
            tmp2_idx = []
            tmp2_type = []
            for exc_indicies, exc_type in zip(tmp1_idx,tmp1_type):
                if i in exc_indicies:
                    new_excitation_indicies.append(exc_indicies)
                    new_excitation_operator_type.append(exc_type)
                else:
                    tmp2_idx.append(exc_indicies)
                    tmp2_type.append(exc_type)
            tmp1_idx = copy.deepcopy(tmp2_idx)
            tmp1_type = copy.deepcopy(tmp2_type)
        self.excitation_indicies = copy.deepcopy(new_excitation_indicies)
        self.excitation_operator_type = copy.deepcopy(new_excitation_operator_type)

    def add_singles(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add singles.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i in iterate_t1(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, a))
            self.excitation_operator_type.append("single")
            self.n_params += 1

    def add_doubles(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add doubles.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, in iterate_t2(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, a, b))
            self.excitation_operator_type.append("double")
            self.n_params += 1

    def add_pair_doubles(self, active_occ_idx: Sequence[int], active_unocc_idx: Sequence[int]) -> None:
        """Add doubles.

        Args:
            active_occ_idx: Active strongly occupied spatial orbital indices.
            active_unocc_idx: Active weakly occupied spatial orbital indices.
        """
        for a, i, b, j, in iterate_t2(active_occ_idx, active_unocc_idx):
            self.excitation_indicies.append((i, j, a, b))
            self.excitation_operator_type.append("double")
            self.n_params += 1

    def add_triples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving triples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k in iterate_t3(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, a, b, c))
            self.excitation_operator_type.append("triple")
            self.n_params += 1

    def add_quadruples(
        self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]
    ) -> None:
        """Add alpha-number and beta-number conserving quadruples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l in iterate_t4(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, a, b, c, d))
            self.excitation_operator_type.append("quadruple")
            self.n_params += 1

    def add_quintuples(
        self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]
    ) -> None:
        """Add alpha-number and beta-number conserving quintuples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l, e, m in iterate_t5(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, m, a, b, c, d, e))
            self.excitation_operator_type.append("quintuple")
            self.n_params += 1

    def add_sextuples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving sextuples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, m, n, a, b, c, d, e, f))
            self.excitation_operator_type.append("sextuple")
            self.n_params += 1

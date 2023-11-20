import numpy as np


class ReducedDenstiyMatrix:
    def __init__(
        self,
        num_inactive_orbs: int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        rdm1: np.ndarray,
        rdm2: np.ndarray | None = None,
        rdm3: np.ndarray | None = None,
    ) -> None:
        self.inactive_idx = []
        self.active_idx = []
        self.virtual_idx = []
        for idx in range(num_inactive_orbs):
            self.inactive_idx.append(idx)
        for idx in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            self.active_idx.append(idx)
        for idx in range(
            num_inactive_orbs + num_active_orbs,
            num_inactive_orbs + num_active_orbs + num_virtual_orbs,
        ):
            self.virtual_idx.append(idx)
        self.idx_shift = num_inactive_orbs
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.rdm3 = rdm3

    def RDM1(self, p: int, q: int) -> float:
        if p in self.active_idx and q in self.active_idx:
            return self.rdm1[p - self.idx_shift, q - self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 2
            return 0
        return 0

    def RDM2(self, p: int, q: int, r: int, s: int) -> float:
        if self.rdm2 is None:
            raise ValueError("RDM2 is not given.")
        if p in self.active_idx and q in self.active_idx and r in self.active_idx and s in self.active_idx:
            return self.rdm2[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1[q - self.idx_shift, r - self.idx_shift]
            return 0
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
        ):
            if q == r:
                return -self.rdm1[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return 2 * self.rdm1[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.active_idx
        ):
            if p == q:
                return 2 * self.rdm1[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            val = 0
            if p == q and r == s:
                val += 4
            if q == r and p == s:
                val -= 2
            return val
        return 0

    def RDM3(self, p: int, q: int, r: int, s: int, t: int, u: int) -> float:
        r"""Get three-electron reduced density matrix element.

        .. math::
            \Gamma^{[3]}_{pqrstu} = \left\{\begin{array}{ll}
                8\delta_{ij}\delta_{kl}\delta_{mn} - 4\delta_{lm}\delta_{ij}\delta_{kn} - 4\delta_{jk}\delta_{il}\delta_{mn} - 4\delta_{jm}\delta_{in}\delta_{kl} + 2\delta_{jk}\delta_{lm}\delta_{in} + 2\delta_{jm}\delta_{kn}\delta_{il} & pqrstu = ijklmn \\
                4 \delta_{ij}\delta_{kl}\Gamma^{[1]}_{vw} - 2\delta_{jk}\delta_{il}\Gamma^{[1]}_{vw} & pqrstu = vwijkl \\
                \delta_{il}\delta_{kj}\Gamma^{[1]}_{wv} - 2\delta_{ij}\delta_{kl}\Gamma^{[1]}_{wv} & pqrstu = ivwjkl \\
                2\delta_{ij}\Gamma^{[2]}_{vwxy} & pqrstu = vwxyij \\
                - \delta_{ij}\Gamma^{[2]}_{vwyx} & pqrstu = vwixyj \\
                \left<0\left|\hat{E}^{wya}_{vxz}\right|0\right> & pqrstu = vwxyza \\
                0 & \text{otherwise} \\
                \end{array} \right.

        and the symmetry `\Gamma^{[3]}_{pqrstu}=\Gamma^{[3]}_{pqturs}=\Gamma^{[3]}_{rspqtu}=\Gamma^{[3]}_{rstupq}=\Gamma^{[3]}_{tupqrs}=\Gamma^{[3]}_{turspq}=\Gamma^{[3]}_{qpsrut}=\Gamma^{[3]}_{qputsr}=\Gamma^{[3]}_{srqput}=\Gamma^{[3]}_{srutqp}=\Gamma^{[3]}_{utqpsr}=\Gamma^{[3]}_{utsrqp}`:math:.

        Args:
            p: Spatial orbital index.
            q: Spatial orbital index.
            r: Spatial orbital index.
            s: Spatial orbital index.
            t: Spatial orbital index.
            u: Spatial orbital index.

        Returns:
            Three-electron reduced density matrix element.
        """
        if self.rdm2 is None:
            raise ValueError("RDM2 is not given.")
        if self.rdm3 is None:
            raise ValueError("RDM3 is not given.")
        # PQ RS TU
        # A = Active, I = Inactive
        # AA AA AA
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            return self.rdm3[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
                t - self.idx_shift,
                u - self.idx_shift,
            ]
        # AA AI IA
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.active_idx
        ):
            val = 0
            if t == s:
                val -= self.rdm2[
                    p - self.idx_shift,
                    q - self.idx_shift,
                    r - self.idx_shift,
                    u - self.idx_shift,
                ]
            return val
        # AA IA AI
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.inactive_idx
        ):
            val = 0
            if r == u:
                val -= self.rdm2[
                    p - self.idx_shift,
                    q - self.idx_shift,
                    t - self.idx_shift,
                    s - self.idx_shift,
                ]
            return val
        # AI AA IA
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.inactive_idx
            and u in self.active_idx
        ):
            val = 0
            if q == t:
                val -= self.rdm2[
                    p - self.idx_shift,
                    u - self.idx_shift,
                    r - self.idx_shift,
                    s - self.idx_shift,
                ]
            return val
        # IA AA AI
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == u:
                val -= self.rdm2[
                    t - self.idx_shift,
                    q - self.idx_shift,
                    r - self.idx_shift,
                    s - self.idx_shift,
                ]
            return val
        # AI IA AA
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            val = 0
            if q == r:
                val -= self.rdm2[
                    p - self.idx_shift,
                    s - self.idx_shift,
                    t - self.idx_shift,
                    u - self.idx_shift,
                ]
            return val
        # IA AI AA
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            val = 0
            if p == s:
                val -= self.rdm2[
                    r - self.idx_shift,
                    q - self.idx_shift,
                    t - self.idx_shift,
                    u - self.idx_shift,
                ]
            return val
        # AA AA II
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if t == u:
                val += (
                    2
                    * self.rdm2[
                        p - self.idx_shift,
                        q - self.idx_shift,
                        r - self.idx_shift,
                        s - self.idx_shift,
                    ]
                )
            return val
        # AA II AA
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            val = 0
            if r == s:
                val += (
                    2
                    * self.rdm2[
                        p - self.idx_shift,
                        q - self.idx_shift,
                        t - self.idx_shift,
                        u - self.idx_shift,
                    ]
                )
            return val
        # II AA AA
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            val = 0
            if p == q:
                val += (
                    2
                    * self.rdm2[
                        r - self.idx_shift,
                        s - self.idx_shift,
                        t - self.idx_shift,
                        u - self.idx_shift,
                    ]
                )
            return val
        # AI IA II
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if t == q and r == u:
                val += self.rdm1[p - self.idx_shift, s - self.idx_shift]
            if r == q and t == u:
                val -= 2 * self.rdm1[p - self.idx_shift, s - self.idx_shift]
            return val
        # AI II IA
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.active_idx
        ):
            val = 0
            if r == q and t == s:
                val += self.rdm1[p - self.idx_shift, u - self.idx_shift]
            if t == q and r == s:
                val -= 2 * self.rdm1[p - self.idx_shift, u - self.idx_shift]
            return val
        # IA AI II
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == u and t == s:
                val += self.rdm1[q - self.idx_shift, r - self.idx_shift]
            if p == s and t == u:
                val -= 2 * self.rdm1[q - self.idx_shift, r - self.idx_shift]
            return val
        # II AI IA
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.active_idx
        ):
            val = 0
            if p == s and t == q:
                val += self.rdm1[r - self.idx_shift, u - self.idx_shift]
            if p == q and t == s:
                val -= 2 * self.rdm1[r - self.idx_shift, u - self.idx_shift]
            return val
        # IA II AI
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.active_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == s and r == u:
                val += self.rdm1[q - self.idx_shift, t - self.idx_shift]
            if p == u and r == s:
                val -= 2 * self.rdm1[q - self.idx_shift, t - self.idx_shift]
            return val
        # II IA AI
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
            and t in self.active_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == u and r == q:
                val += self.rdm1[s - self.idx_shift, t - self.idx_shift]
            if p == q and r == u:
                val -= 2 * self.rdm1[s - self.idx_shift, t - self.idx_shift]
            return val
        # AA II II
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if r == s and t == u:
                val += 4 * self.rdm1[p - self.idx_shift, q - self.idx_shift]
            if r == u and t == s:
                val -= 2 * self.rdm1[p - self.idx_shift, q - self.idx_shift]
            return val
        # II AA II
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.active_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == q and t == u:
                val += 4 * self.rdm1[r - self.idx_shift, s - self.idx_shift]
            if p == u and t == q:
                val -= 2 * self.rdm1[r - self.idx_shift, s - self.idx_shift]
            return val
        # II II AA
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.active_idx
            and u in self.active_idx
        ):
            val = 0
            if p == q and r == s:
                val += 4 * self.rdm1[t - self.idx_shift, u - self.idx_shift]
            if p == s and r == q:
                val -= 2 * self.rdm1[t - self.idx_shift, u - self.idx_shift]
            return val
        # II II II
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
            and t in self.inactive_idx
            and u in self.inactive_idx
        ):
            val = 0
            if p == q and r == s and t == u:
                val += 8
            if s == t and p == q and r == u:
                val -= 4
            if q == r and p == s and t == u:
                val -= 4
            if q == t and p == u and r == s:
                val -= 4
            if q == r and s == t and p == u:
                val += 2
            if q == t and r == u and p == s:
                val += 2
            return val
        return 0


def get_orbital_gradient(
    rdms: ReducedDenstiyMatrix,
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    """ """
    gradient = np.zeros(len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += 2 * h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= 2 * h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return gradient


def get_orbital_gradient_response(
    rdms: ReducedDenstiyMatrix,
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    """ """
    gradient = np.zeros(2 * len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += 1 / 2 * g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= 1 / 2 * g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= 1 / 2 * g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += 1 / 2 * g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    shift = len(kappa_idx)
    for idx, (n, m) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx + shift] += h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx + shift] -= h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx + shift] += 1 / 2 * g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx + shift] -= 1 / 2 * g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx + shift] -= 1 / 2 * g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx + shift] += 1 / 2 * g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return 2 ** (-1 / 2) * gradient


def get_orbital_response_metric_sgima(rdms: ReducedDenstiyMatrix, kappa_idx: list[list[int]]) -> np.ndarray:
    sigma = np.zeros((len(kappa_idx), len(kappa_idx)))
    for idx1, (n, m) in enumerate(kappa_idx):
        for idx2, (p, q) in enumerate(kappa_idx):
            if p == n:
                sigma[idx1, idx2] += rdms.RDM1(m, q)
            if m == q:
                sigma[idx1, idx2] -= rdms.RDM1(p, n)
    return -1 / 2 * sigma


def get_orbital_response_vector_norm(
    rdms: ReducedDenstiyMatrix,
    kappa_idx: list[list[int]],
    response_vectors: np.ndarray,
    state_number: int,
    number_excitations: int,
) -> float:
    norm = 0
    for i, (m, n) in enumerate(kappa_idx):
        for j, (mp, np) in enumerate(kappa_idx):
            if n == np:
                norm += (
                    response_vectors[i, state_number] * response_vectors[j, state_number] * rdms.RDM1(m, mp)
                )
            if m == mp:
                norm -= (
                    response_vectors[i, state_number] * response_vectors[j, state_number] * rdms.RDM1(n, np)
                )
            if m == mp:
                norm += (
                    response_vectors[i + number_excitations, state_number]
                    * response_vectors[j + number_excitations, state_number]
                    * rdms.RDM1(n, np)
                )
            if n == np:
                norm -= (
                    response_vectors[i + number_excitations, state_number]
                    * response_vectors[j + number_excitations, state_number]
                    * rdms.RDM1(m, mp)
                )
    return 1 / 2 * norm


def get_orbital_response_property_gradient(
    rdms: ReducedDenstiyMatrix,
    x_mo: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    response_vectors: np.ndarray,
    state_number: int,
    number_excitations: int,
) -> float:
    """ """
    prop_grad = 0
    for i, (m, n) in enumerate(kappa_idx):
        for p in range(num_inactive_orbs + num_active_orbs):
            prop_grad += (
                (response_vectors[i + number_excitations, state_number] - response_vectors[i, state_number])
                * x_mo[n, p]
                * rdms.RDM1(m, p)
            )
            prop_grad += (
                (response_vectors[i, state_number] - response_vectors[i + number_excitations, state_number])
                * x_mo[m, p]
                * rdms.RDM1(n, p)
            )
    return 2 ** (-1 / 2) * prop_grad


def get_orbital_response_hessian_block(
    rdms: ReducedDenstiyMatrix,
    h: np.ndarray,
    g: np.ndarray,
    kappa_idx1: list[list[int]],
    kappa_idx2: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    A1e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    A2e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            # 1e contribution
            A1e[idx1, idx2] += h[n, t] * rdms.RDM1(m, u)
            A1e[idx1, idx2] += h[u, m] * rdms.RDM1(t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u:
                    A1e[idx1, idx2] -= h[n, p] * rdms.RDM1(t, p)
                if t == n:
                    A1e[idx1, idx2] -= h[p, m] * rdms.RDM1(p, u)
            # 2e contribution
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    A2e[idx1, idx2] += g[n, t, p, q] * rdms.RDM2(m, u, p, q)
                    A2e[idx1, idx2] -= g[n, p, u, q] * rdms.RDM2(m, p, t, q)
                    A2e[idx1, idx2] += g[n, p, q, t] * rdms.RDM2(m, p, q, u)
                    A2e[idx1, idx2] += g[u, m, p, q] * rdms.RDM2(t, n, p, q)
                    A2e[idx1, idx2] += g[p, m, u, q] * rdms.RDM2(p, n, t, q)
                    A2e[idx1, idx2] -= g[p, m, q, t] * rdms.RDM2(p, n, q, u)
                    A2e[idx1, idx2] -= g[u, p, n, q] * rdms.RDM2(t, p, m, q)
                    A2e[idx1, idx2] += g[p, t, n, q] * rdms.RDM2(p, u, m, q)
                    A2e[idx1, idx2] += g[p, q, n, t] * rdms.RDM2(p, q, m, u)
                    A2e[idx1, idx2] += g[u, p, q, m] * rdms.RDM2(t, p, q, n)
                    A2e[idx1, idx2] -= g[p, t, q, m] * rdms.RDM2(p, u, q, n)
                    A2e[idx1, idx2] += g[p, q, u, m] * rdms.RDM2(p, q, t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g[n, p, q, r] * rdms.RDM2(t, p, q, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, m, q, r] * rdms.RDM2(p, u, q, r)
                        if m == u:
                            A2e[idx1, idx2] -= g[p, q, n, r] * rdms.RDM2(p, q, t, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, q, r, m] * rdms.RDM2(p, q, r, u)
    return 1 / 2 * A1e + 1 / 4 * A2e


def get_electronic_energy(
    rdms: ReducedDenstiyMatrix,
    h: np.ndarray,
    g: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> float:
    r"""Calculate electronic energy.

    .. math::
        E = \left<0\left|\hat{H}\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       h: One-electron integrals in MO in Hamiltonian.
       g: Two-electron integrals in MO in Hamiltonian.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        The electronic energy.
    """
    energy = 0
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            energy += h[p, q] * rdms.RDM1(p, q)
            for r in range(num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs + num_active_orbs):
                    energy += 1 / 2 * g[p, q, r, s] * rdms.RDM2(p, q, r, s)
    return energy


def get_projected_orbital_response_hessian_block(
    rdms: ReducedDenstiyMatrix,
    h: np.ndarray,
    g: np.ndarray,
    kappa_idx1: list[list[int]],
    kappa_idx2: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    A1e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    A2e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    # A = get_orbital_response_hessian_block(rdms, h, g, kappa_idx1, kappa_idx2, num_inactive_orbs, num_active_orbs)
    # print("")
    # print("proj-q Naive part")
    # with np.printoptions(precision=3, suppress=True):
    #    print(A)
    A = get_orbital_response_metric_sgima(rdms, kappa_idx1) * get_electronic_energy(
        rdms, h, g, num_inactive_orbs, num_active_orbs
    )
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            # 1e contribution
            A1e[idx1, idx2] -= h[n, t] * rdms.RDM1(m, u)
            for p in range(num_inactive_orbs + num_active_orbs):
                A1e[idx1, idx2] += h[u, p] * rdms.RDM2(m, n, t, p)
                A1e[idx1, idx2] -= h[p, t] * rdms.RDM2(m, n, p, u)
                if m == u:
                    A1e[idx1, idx2] += h[n, p] * rdms.RDM1(t, p)
                for q in range(num_inactive_orbs + num_active_orbs):
                    if m == u:
                        A1e[idx1, idx2] += h[p, q] * rdms.RDM2(t, n, p, q)
                    if t == n:
                        A1e[idx1, idx2] -= h[p, q] * rdms.RDM2(m, u, p, q)
            # 2e contribution
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    A2e[idx1, idx2] -= g[u, p, n, q] * rdms.RDM2(m, q, t, p)
                    A2e[idx1, idx2] += g[n, t, p, q] * rdms.RDM2(m, u, p, q)
                    A2e[idx1, idx2] += g[p, t, n, q] * rdms.RDM2(m, q, p, u)
                    A2e[idx1, idx2] -= g[n, p, u, q] * rdms.RDM2(m, p, t, q)
                    A2e[idx1, idx2] += g[n, p, q, t] * rdms.RDM2(m, p, q, u)
                    A2e[idx1, idx2] += g[p, q, n, t] * rdms.RDM2(m, u, p, q)
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if t == n:
                            A2e[idx1, idx2] += g[u, p, q, r] * rdms.RDM2(m, p, q, r)
                            A2e[idx1, idx2] += g[p, q, u, r] * rdms.RDM2(m, r, p, q)
                            A2e[idx1, idx2] -= g[u, p, q, r] * rdms.RDM2(m, p, q, r)
                            A2e[idx1, idx2] -= g[p, q, u, r] * rdms.RDM2(m, r, p, q)
                        if m == u:
                            A2e[idx1, idx2] -= g[n, p, q, r] * rdms.RDM2(t, p, q, r)
                            A2e[idx1, idx2] -= g[p, q, n, r] * rdms.RDM2(t, r, p, q)
                        for s in range(num_inactive_orbs + num_active_orbs):
                            A2e[idx1, idx2] -= g[u, p, q, r] * rdms.RDM3(m, n, t, p, q, r)
                            A2e[idx1, idx2] += g[p, t, q, r] * rdms.RDM3(m, n, p, u, q, r)
                            A2e[idx1, idx2] -= g[p, q, u, r] * rdms.RDM3(m, n, p, q, t, r)
                            A2e[idx1, idx2] += g[p, q, r, t] * rdms.RDM3(m, n, p, q, r, u)
                            if t == n:
                                A2e[idx1, idx2] += g[p, q, r, s] * rdms.RDM3(m, u, p, q, r, s)
                            if m == u:
                                A2e[idx1, idx2] -= g[p, q, r, s] * rdms.RDM3(t, n, p, q, r, s)
    print("")
    print("q-proj A")
    with np.printoptions(precision=3, suppress=True):
        print(A)
    print("")
    print("q-proj correction")
    with np.printoptions(precision=3, suppress=True):
        print(1 / 2 * A1e + 1 / 4 * A2e)
    return A - 1 / 2 * A1e + 1 / 4 * A2e

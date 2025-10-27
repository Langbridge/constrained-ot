import numpy as np
import matplotlib.pyplot as plt
from types import NoneType
from tqdm import tqdm

from scipy.special import logsumexp

class BasicSolver():
    """
    Base class for Sinkhorn-type solvers for OT problems.
    """
    def __init__(self):
        pass

    def solve(self, a, b, mu, nu):
        pass

class SoftBregman(BasicSolver):
    """
    Bregman iteration solver to solve the Relaxed OT problem, where certain transport pairings are impossible (blocked_idxs)
    and certain source and target indices are not strictly constrained (rows_ and cols_to_relax).
    """

    def __init__(self, gamma, gamma_c, gamma_r, rows_to_relax=None, cols_to_relax=None, blocked_idxs=None):
        """

        Args:
            gamma (float): Regularisation parameter for the kernel.
            gamma_c (float): Penalisation parameter for the column constraints.
            gamma_r (float): Penalisation parameter for the row constraints.
            rows_to_relax (np.array, optional): Array of indices of the source to relax. Defaults to None.
            cols_to_relax (np.array, optional): Array of indices of the target to relax. Defaults to None.
            blocked_idxs (tuple, optional): Tuple of index pairs that are blocked in the transport plan. Defaults to None.
        """

        self.gamma = gamma

        if type(rows_to_relax) == NoneType:
            self.row_exponent_0, self.row_exponent_1 = 1., 0.
        else:
            self.row_exponent_0, self.row_exponent_1 = self._prep_exponents(gamma_r, rows_to_relax)

        if type(cols_to_relax) == NoneType:
            self.col_exponent_0, self.col_exponent_1 = 1., 0.
        else:
            self.col_exponent_0, self.col_exponent_1 = self._prep_exponents(gamma_c, cols_to_relax)

        if type(blocked_idxs) == NoneType:
            self.blocked_idxs = ()
        else:
            self.blocked_idxs = blocked_idxs

    def _prep_exponents(self, gamma, relax_idx):
        exponent_0 = gamma/(1+gamma) * relax_idx
        exponent_0[relax_idx == 0] = 1.0
        exponent_1 = relax_idx * (-1/gamma) # 0s are implicit here

        return exponent_0, exponent_1

    def solve(self, a, b, mu, nu, C=None, num_iters=int(1e3), plot=False, thresh=0.0):
        """_summary_

        Args:
            a (np.array): Locations of the source mass.
            b (np.array): Locations of the target mass.
            mu (np.array): Source masses.
            nu (np.array): Target masses.
            num_iters (int, optional): Number of iterations to solve. Defaults to int(1e3).
            plot (bool, optional): Plots convergence graph in log space if True. Defaults to False.
            thresh (float, optional): Threshold for early stopping. Defaults to 0.0 (i.e. no early stopping).

        Returns:
            T (np.array): Transport plan.
        """
        if type(C) == NoneType:
            C = np.random.random(size=(len(a), len(b))) ** 2

        C /= np.max(C) # numerical stability

        K = np.pow(np.e, -C / self.gamma)
        K[*self.blocked_idxs.T] = 0.0

        # K[K < 1e-300] = 1e-300  # Avoid underflow
        K[K > 1e300] = 1e300    # Avoid overflow

        mu_0, nu_0 = mu.copy(), nu.copy()

        T = K.copy()
        T_old = K.copy()
        convergence_list = np.zeros(shape=(num_iters))

        for i in tqdm(range(num_iters)):
            D1 = np.divide(mu_0[:,None], np.sum(T, axis=1, keepdims=True), out=np.ones_like(mu_0[:,None]), where=(np.sum(T, axis=1, keepdims=True) != 0.0))
            D1 = np.pow(D1, self.row_exponent_0)
            T = D1 * T
            mu = (np.pow(D1, self.row_exponent_1) * mu[:,None]).reshape(-1,)

            D2 = np.divide(nu_0[None,:], np.sum(T, axis=0, keepdims=True), out=np.ones_like(nu_0[None,:]), where=(np.sum(T, axis=0, keepdims=True) != 0.0))
            D2 = np.pow(D2, self.col_exponent_0)
            T = D2 * T
            nu = (np.pow(D2, self.col_exponent_1) * nu[None,:]).reshape(-1,)

            convergence_list[i] = np.linalg.norm((T - T_old).ravel(), ord=1)
            T_old = T.copy()

            if convergence_list[i] < thresh:
                break

        if plot:
            fig, ax = plt.subplots(dpi=100)
            ax.plot(convergence_list[:i] / convergence_list[0], c='k')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Change in T')

        return T
    
class LogStabilised_SoftBregman(SoftBregman):
    def __init__(self, gamma, gamma_c, gamma_r, rows_to_relax=None, cols_to_relax=None, blocked_idxs=None):
        super().__init__(gamma, gamma_c, gamma_r, rows_to_relax, cols_to_relax, blocked_idxs)

    def solve(self, a, b, mu, nu, C=None, num_iters=int(1e3), plot=False):
        """_summary_

        Args:
            a (np.array): Locations of the source mass.
            b (np.array): Locations of the target mass.
            mu (np.array): Source masses.
            nu (np.array): Target masses.
            num_iters (int, optional): Number of iterations to solve. Defaults to int(1e3).
            plot (bool, optional): Plots convergence graph in log space if True. Defaults to False.

        Returns:
            T (np.array): Transport plan.
        """
        if type(C) == NoneType:
            C = np.random.random(size=(len(a), len(b))) ** 2

        C /= np.sum(C)
        logK = - C / self.gamma
        CLIP_MIN = -1000.0
        logK = np.maximum(logK, logK.max() + CLIP_MIN)

        # maskK = np.zeros_like(logK)
        # for (x,y) in self.blocked_idxs:
        #     maskK[x, y] = 1
        # logK = np.ma.masked_array(logK, mask=maskK)

        log_mu0 = np.log(mu)
        log_nu0 = np.log(nu)
        log_mu = np.zeros_like(log_mu0)
        log_nu = np.zeros_like(log_nu0)

        logT = logK.copy()
        logT_old = logK.copy()
        convergence_list = np.zeros(shape=(num_iters))

        print("C min/mean/max:", float(C.min()), float(C.mean()), float(C.max()))
        print("gamma (epsilon) value:", float(self.gamma))
        print("logK min/max/mean/std:", float(logK.min()), float(logK.max()), float(logK.mean()), float(logK.std()))
        print("Are any mu or nu zero or extremely tiny?")
        print(" mu min, max, sum:", float(mu.min()), float(mu.max()), float(mu.sum()))
        print(" nu min, max, sum:", float(nu.min()), float(nu.max()), float(nu.sum()))


        for i in tqdm(range(num_iters)):
            # row scaling
            logD1 = self.row_exponent_0 * (log_mu0 - logsumexp(logT, axis=1))[:,None]
            logT += logD1
            log_mu += (self.row_exponent_1 * logD1).reshape(-1,)

            # col scaling
            logD2 = self.col_exponent_0 * (log_nu0 - logsumexp(logT, axis=0))[None,:]
            logT += logD2
            log_nu += (self.col_exponent_1 * logD2).reshape(-1,)

            # stabilise log variables
            logT -= np.mean(logT)
            log_mu -= np.mean(log_mu)
            log_nu -= np.mean(log_nu)

            # convergence testing
            convergence_list[i] = np.linalg.norm((logT - logT_old).ravel(), ord=1)
            logT_old = logT.copy()

        if plot:
            fig, ax = plt.subplots(dpi=100)
            ax.plot(convergence_list / convergence_list[0], c='k')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Log Change in T')

        # logT = logT.filled(0.0)
        T = np.exp(logT - logsumexp(logT)) # avoid overflow
        mu = np.exp(log_mu - logsumexp(log_mu))
        nu = np.exp(log_nu - logsumexp(log_nu))

        diagnose(logK, logT, mu, nu)

        return T

class BoundedDykstra(BasicSolver):
    """
    Solver based on Dykstra's algorithm to solve OT problems where the constraints are non-affine,
    such as inequality constraints which arise when the source and target are bounded.
    """
    def __init__(self, gamma, row_ubs, row_lbs, col_ubs, col_lbs, blocked_idxs=None):
        """_summary_

        Args:
            gamma (_type_): Regularisation parameter for the kernel.
            row_ubs (_type_): List or np.array of upper bounds for the source elements. To strictly bound elements, set the ub and lb to the original mass.
            row_lbs (_type_): List or np.array of lower bounds for the source elements. To strictly bound elements, set the ub and lb to the original mass.
            col_ubs (_type_): List or np.array of upper bounds for the target elements. To strictly bound elements, set the ub and lb to the original mass.
            col_lbs (_type_): List or np.array of lower bounds for the target elements. To strictly bound elements, set the ub and lb to the original mass.
            blocked_idxs (_type_, optional): _description_. Tuple of index pairs that are blocked in the transport plan. Defaults to None.
        """

        self.gamma = gamma

        self.row_bounds = (row_ubs, row_lbs)
        self.col_bounds = (col_ubs, col_lbs)

        if type(blocked_idxs) == NoneType:
            self.blocked_idxs = ()
        else:
            self.blocked_idxs = blocked_idxs

    def _C_row_upper(self, T, q, mu, p_max): # upper bound
        t = T * q
        mu = np.minimum(np.sum(T, axis=1), p_max) # scale to either remain the same or push to bound
        T = t * np.divide(mu[:,None], np.sum(t, axis=1, keepdims=True))
        q = np.divide(t, T, out=np.ones_like(T), where=(T != 0.0)) # adjust for zeros in the plan
        return T, q, mu

    def _C_row_lower(self, T, q, mu, p_min): # lower bound
        t = T * q
        mu = np.maximum(np.sum(T, axis=1), p_min) # scale to either remain the same or push to bound
        T = t * np.divide(mu[:,None], np.sum(t, axis=1, keepdims=True))
        q = np.divide(t, T, out=np.ones_like(T), where=(T != 0.0)) # adjust for zeros in the plan
        return T, q, mu

    def _C_col_upper(self, T, q, nu, p_max): # upper bound
        t = T * q
        nu = np.minimum(np.sum(T, axis=0), p_max) # scale to either remain the same or push to bound
        T = t * np.divide(nu[None,:], np.sum(t, axis=0, keepdims=True))
        q = np.divide(t, T, out=np.ones_like(T), where=(T != 0.0))
        return T, q, nu

    def _C_col_lower(self, T, q, nu, p_min): # lower bound
        t = T * q
        nu = np.maximum(np.sum(T, axis=0), p_min) # scale to either remain the same or push to bound
        T = t * np.divide(nu[None,:], np.sum(t, axis=0, keepdims=True))
        q = np.divide(t, T, out=np.ones_like(T), where=(T != 0.0))
        return T, q, nu
    
    def solve(self, a, b, mu, nu, C=None, num_iters=int(1e3), plot=False):
        """_summary_

        Args:
            a (np.array): Locations of the source mass.
            b (np.array): Locations of the target mass.
            mu (np.array): Source masses.
            nu (np.array): Target masses.
            num_iters (int, optional): Number of iterations to solve. Defaults to int(1e3).
            plot (bool, optional): Plots convergence graph in log space if True. Defaults to False.

        Returns:
            T (np.array): Transport plan.
        """
        if type(C) == NoneType:
            C = np.random.random(size=(len(a), len(b))) ** 2
        K = np.pow(np.e, -C / self.gamma)
        for (x,y) in self.blocked_idxs:
            K[x, y] = 0

        T = K.copy()
        T_old = K.copy()

        # 'activity' switches for non-affine constraints
        q0 = np.ones_like(K)
        q1 = np.ones_like(K)
        q2 = np.ones_like(K)
        q3 = np.ones_like(K)

        convergence_list = np.zeros(shape=(num_iters))

        for i in range(num_iters):
            # row constraints
            T, q0, mu = self._C_row_upper(T, q0, mu, self.row_bounds[0]) # ub
            T, q1, mu = self._C_row_lower(T, q1, mu, self.row_bounds[1]) # lb

            # col constraints
            T, q2, nu = self._C_col_upper(T, q2, nu, self.col_bounds[0]) # ub
            T, q3, nu = self._C_col_lower(T, q3, nu, self.col_bounds[1]) # lb

            convergence_list[i] = np.linalg.norm((T - T_old).ravel(), ord=1)
            T_old = T.copy()

        if plot:
            fig, ax = plt.subplots()
            ax.plot(convergence_list / convergence_list[0], c='k')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Change in T')

        return T
        

def diagnose(logK, logT, mu, nu):
    tiny = 1e-300
    print("shapes: logK, logT =", logK.shape, logT.shape)
    print("any nan/inf in logT?", np.isnan(logT).any(), np.isinf(logT).any())
    print("logT max/min/mean/std:", float(logT.max()), float(logT.min()), float(logT.mean()), float(logT.std()))
    # normalized log-partition function
    logZ = logsumexp(logT)
    print("logZ (global):", float(logZ))
    # safe normalized T in log-space
    logT_norm = logT - logZ
    # check sum in log-space via logsumexp -> should be 0
    print("logsumexp(logT_norm) (should be 0):", float(logsumexp(logT_norm)))
    # compute T sum (exact)
    T_sum = float(np.exp(logsumexp(logT)))
    print("exp(logZ) i.e. sum(exp(logT)):", T_sum)
    # compute marginals from logT without full exp (safe)
    approx_row = np.exp(logsumexp(logT, axis=1))
    approx_col = np.exp(logsumexp(logT, axis=0))
    print("||row_sum - mu||:", np.linalg.norm(approx_row - mu))
    print("||col_sum - nu||:", np.linalg.norm(approx_col - nu))
    # entropy computed in log-space (safe)
    # H = - sum_i p_i log p_i = - sum_i exp(logT_norm_i) * logT_norm_i
    T_flat = np.exp(logT_norm)   # may be large array but safe because normalized by logZ
    H = -np.sum(T_flat * logT_norm)
    print("entropy (safe computed):", float(H))
    print("entropy upper bound (uniform) log(n*m):", np.log(logT.size))
    # check mu/nu logs
    print("mu min/max, any zeros?:", float(np.min(mu)), float(np.max(mu)))
    print("nu min/max, any zeros?:", float(np.min(nu)), float(np.max(nu)))


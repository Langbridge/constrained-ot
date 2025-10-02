import numpy as np
import matplotlib.pyplot as plt
from types import NoneType

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
            self.row_exponent_0 = gamma_r / (1+gamma_r)
            self.row_exponent_1 = -1 / gamma_r
        else:
            self.row_exponent_0, self.row_exponent_1 = self._prep_exponents(gamma_r, rows_to_relax)

        if type(cols_to_relax) == NoneType:
            self.row_exponent_0 = gamma_r / (1+gamma_r)
            self.row_exponent_1 = -1 / gamma_r
        else:
            self.col_exponent_0, self.col_exponent_1 = self._prep_exponents(gamma_c, cols_to_relax)

        if type(blocked_idxs) == NoneType:
            self.blocked_idxs = ()
        else:
            self.blocked_idxs = blocked_idxs

    def _prep_exponents(self, gamma, relax_idx):
        exponent_0 = gamma/(1+gamma) * relax_idx
        exponent_0[relax_idx == 0] = 1.0
        exponent_1 = relax_idx * (-1/gamma) # 1s are implicit here

        return exponent_0, exponent_1

    def solve(self, a, b, mu, nu, num_iters=int(1e3), plot=False):
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
        C = np.random.random(size=(len(a), len(b))) ** 2
        K = np.pow(np.e, -C / self.gamma)
        for (x,y) in self.blocked_idxs:
            K[x, y] = 0

        T = K.copy()
        T_old = K.copy()
        convergence_list = np.zeros(shape=(num_iters))

        for i in range(num_iters):
            D1 = np.divide(mu[:,None], np.sum(T, axis=1, keepdims=True), out=np.ones_like(mu[:,None]), where=(np.sum(T, axis=1, keepdims=True) != 0.0))
            D1 = np.pow(D1, self.row_exponent_0)
            T = D1 * T
            mu = (np.pow(D1, self.row_exponent_1) * mu[:,None]).reshape(-1,)

            D2 = np.divide(nu[None,:], np.sum(T, axis=0, keepdims=True), out=np.ones_like(nu[None,:]), where=(np.sum(T, axis=0, keepdims=True) != 0.0))
            D2 = np.pow(D2, self.col_exponent_0)
            T = D2 * T
            nu = (np.pow(D2, self.col_exponent_1) * nu[None,:]).reshape(-1,)

            convergence_list[i] = np.linalg.norm((T - T_old).ravel(), ord=1)
            T_old = T.copy()

        if plot:
            fig, ax = plt.subplots()
            ax.plot(convergence_list / convergence_list[0], c='k')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Change in T')

        return T
    
class BoundedDykstra(BasicSolver):
    """
    Solver based on Dykstra's algorithm to solve OT problems where the constraints are non-affine,
    such as inequality constraints which arise when the source and target are bounded.
    """
    def __init__(self, gamma, row_ubs, row_lbs, col_ubs, col_lbs, blocked_idxs=None):
        """_summary_

        Args:
            gamma (_type_): _description_
            row_ubs (_type_): _description_
            row_lbs (_type_): _description_
            col_ubs (_type_): _description_
            col_lbs (_type_): _description_
            blocked_idxs (_type_, optional): _description_. Defaults to None.
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
    
    def solve(self, a, b, mu, nu, num_iters=int(1e3), plot=False):
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
        
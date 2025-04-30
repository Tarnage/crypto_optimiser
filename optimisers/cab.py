# optimisers/cab.py
import numpy as np
from optimisers.base import Optimiser

class CAB(Optimiser):
    """
    Collective Animal Behaviour optimiser.
    Works with the standard ask() / tell() loop so you can
    plug it straight into run_experiment.py.

    Parameters
    ----------
    r_repulsion, r_orientation, r_attraction : float
        Interaction radii (fractions of the search‐span) for
        repulsion, orientation, attraction zones.
    alpha : float
        Pull towards global best.
    beta : float
        Random‑walk magnitude.
    pop_size : int
        Number of agents.
    """

    def __init__(self, obj_fn, bounds,
                 r_repulsion=0.2,
                 r_orientation=0.7,
                 r_attraction=1.0,
                 alpha=0.1, beta=0.05,
                 pop_size=30, seed=0):
        super().__init__(obj_fn, bounds, seed)
        self.pop_size = pop_size
        self.r_rep, self.r_ori, self.r_att = r_repulsion, r_orientation, r_attraction
        self.alpha, self.beta = alpha, beta

        lo, hi   = self.bounds[:, 0], self.bounds[:, 1]
        self.span = hi - lo

        n_dims = self.bounds.shape[0]
        self.pos  = self.rng.uniform(lo, hi, size=(pop_size, n_dims))
        self.fit  = np.full(pop_size, np.inf)

        # global best
        self.gbest_pos = None
        self.gbest_val = np.inf

    # ---------- Optimiser API ----------------------------------
    def ask(self):
        """Return the population to evaluate."""
        return self.pos.tolist()

    def tell(self, thetas, fitnesses):
        """Receive fitnesses, update positions for next generation."""
        self.pos = np.array(thetas)
        self.fit = np.array(fitnesses)

        # ----- Update global best -----
        best_idx = self.fit.argmin()
        if self.fit[best_idx] < self.gbest_val:
            self.gbest_val = self.fit[best_idx]
            self.gbest_pos = self.pos[best_idx].copy()

        # ----- Move agents (CAB rules) -----
        # pre‑compute pairwise Euclidean distances
        dist = np.linalg.norm(
            self.pos[:, None, :] - self.pos[None, :, :],
            axis=-1
        )

        new_pos = self.pos.copy()
        for i in range(self.pop_size):
            rep, ori, att = np.zeros_like(self.pos[i]), np.zeros_like(self.pos[i]), np.zeros_like(self.pos[i])
            ori_count = 0
            for j in range(self.pop_size):
                if i == j: continue
                d = dist[i, j]
                if d == 0:  # exact overlap
                    continue
                rel = (self.pos[j] - self.pos[i]) / d
                if d < self.r_rep * self.span.any():
                    rep -= rel
                elif d < self.r_ori * self.span.any():
                    ori += rel;  ori_count += 1
                elif d < self.r_att * self.span.any():
                    att += rel
            if ori_count:
                ori /= ori_count
            disp = rep + ori + att
            if np.linalg.norm(disp) > 1e-9:
                disp /= np.linalg.norm(disp)

            # global‑best pull
            gb = self.alpha * (self.gbest_pos - self.pos[i]) / (np.linalg.norm(self.gbest_pos - self.pos[i]) + 1e-9)
            # random walk
            rnd = self.beta * (self.rng.random(len(self.bounds)) - 0.5)

            new_pos[i] = self.pos[i] + disp + gb + rnd

        # clip to bounds
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = np.clip(new_pos, lo, hi)

    # convenience for final best
    @property
    def best(self):
        return self.gbest_pos, self.gbest_val

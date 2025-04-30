# optimisers/gwo.py
import numpy as np
from optimisers.base import Optimiser

class GWO(Optimiser):
    """
    Grey Wolf Optimizer (GWO) implemented via ask/tell interface.
    - Maintains alpha, beta, delta as the 3 best agents each gen.
    - a parameter decreases linearly from 2 to 0 over max_gens.
    """
    def __init__(self, obj_fn, bounds, pop_size=30, max_gens=100, seed=0):
        super().__init__(obj_fn, bounds, seed)
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.gen = 0

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        n_dims = self.bounds.shape[0]
        self.pos = self.rng.uniform(lo, hi, size=(pop_size, n_dims))
        self.fit = np.full(pop_size, np.inf)

        # global best across all generations
        self.gbest_pos = None
        self.gbest_val = np.inf

    def ask(self):
        """Return the current population positions for evaluation."""
        return self.pos.tolist()

    def tell(self, thetas, fitnesses):
        """Update population with evaluated fitnesses, then apply GWO update."""
        self.pos = np.array(thetas)
        self.fit = np.array(fitnesses)

        # sort by fitness: lower is better
        idx = np.argsort(self.fit)
        alpha_pos = self.pos[idx[0]].copy()
        alpha_val = self.fit[idx[0]]
        beta_pos  = self.pos[idx[1]].copy()
        beta_val  = self.fit[idx[1]]
        delta_pos = self.pos[idx[2]].copy()
        delta_val = self.fit[idx[2]]

        # update global best
        if alpha_val < self.gbest_val:
            self.gbest_val = alpha_val
            self.gbest_pos = alpha_pos.copy()

        # linearly decreasing coefficient a from 2 → 0
        a = 2 - 2 * (self.gen / float(self.max_gens))

        n_dims = self.bounds.shape[0]
        new_pos = np.empty_like(self.pos)

        for i in range(self.pop_size):
            X1 = np.zeros(n_dims)
            X2 = np.zeros(n_dims)
            X3 = np.zeros(n_dims)
            for d in range(n_dims):
                # alpha influence
                r1, r2 = self.rng.random(), self.rng.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - self.pos[i, d])
                X1[d] = alpha_pos[d] - A1 * D_alpha

                # beta influence
                r1, r2 = self.rng.random(), self.rng.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[d] - self.pos[i, d])
                X2[d] = beta_pos[d] - A2 * D_beta

                # delta influence
                r1, r2 = self.rng.random(), self.rng.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[d] - self.pos[i, d])
                X3[d] = delta_pos[d] - A3 * D_delta

            # new position is average of the three influences
            new_pos[i] = (X1 + X2 + X3) / 3.0

        # clip to bounds
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = np.clip(new_pos, lo, hi)

        # advance generation counter
        self.gen += 1

    # convenience for final best
    @property
    def best(self):
        return self.gbest_pos, self.gbest_val
    
# Example usage:
# from optimisers.gwo import GWO
# opt = GWO(obj_fn, bounds, pop_size=30, max_gens=100, seed=42)
# for _ in range(100):
#     thetas = opt.ask()
#     scores = [obj_fn(t) for t in thetas]
#     opt.tell(thetas, scores)
# best_theta, best_val = opt.gbest_pos, opt.gbest_val

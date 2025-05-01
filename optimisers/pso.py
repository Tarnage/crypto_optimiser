import numpy as np
from optimisers.base import Optimiser

class PSO(Optimiser):
    def __init__(self, obj_fn, bounds, pop_size=30, w=0.7, c1=1.5, c2=1.5, seed=0):
        super().__init__(obj_fn, bounds, seed)
        lo, hi   = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = self.rng.uniform(lo, hi, size=(pop_size, len(bounds)))
        span     = hi - lo
        self.vel = self.rng.uniform(-span, span, self.pos.shape) * 0.1
        self.w, self.c1, self.c2 = w, c1, c2
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full(pop_size, np.inf)
        self.gbest_pos = None
        self.gbest_val = np.inf

    def ask(self):
        return self.pos.tolist()

    def tell(self, thetas, fitnesses):
        f = np.array(fitnesses)
        better = f < self.pbest_val
        self.pbest_val[better] = f[better]
        self.pbest_pos[better] = self.pos[better]
        idx_best = f.argmin()
        if f[idx_best] < self.gbest_val:
            self.gbest_val = f[idx_best]
            self.gbest_pos = self.pos[idx_best].copy()

        r1, r2 = self.rng.random(self.vel.shape), self.rng.random(self.vel.shape)
        cognitive  = self.c1 * r1 * (self.pbest_pos - self.pos)
        social     = self.c2 * r2 * (self.gbest_pos - self.pos)
        self.vel   = self.w * self.vel + cognitive + social
        self.pos  += self.vel

        # clip to bounds
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = np.clip(self.pos, lo, hi)

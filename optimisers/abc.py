# optimisers/abc.py
import numpy as np
from optimisers.base import Optimiser

class ABC(Optimiser):
    """
    Simplified Artificial Bee Colony (ABC) optimiser via ask/tell interface.
    
    - Employed bees exploit existing solutions.
    - Onlooker bees probabilistically choose solutions to exploit.
    - Scout bees explore new solutions when a source is exhausted (no improvement).
    """
    def __init__(self, obj_fn, bounds, pop_size=30, limit=5, seed=0):
        super().__init__(obj_fn, bounds, seed)
        self.pop_size = pop_size
        self.limit    = limit
        self.gen      = 0
        
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.n_dims = self.bounds.shape[0]
        
        # Initialize population
        self.pos   = self.rng.uniform(lo, hi, size=(pop_size, self.n_dims))
        self.fit   = np.full(pop_size, np.inf)
        self.trials= np.zeros(pop_size, dtype=int)
        
        # Global best
        self.gbest_pos = None
        self.gbest_val = np.inf
        
        # For tracing
        self.steps_positions = []

    def ask(self):
        """Return current population for evaluation."""
        return self.pos.tolist()

    def tell(self, thetas, fitnesses):
        """Receive fitnesses, then perform ABC phases to update the population."""
        # Load evaluated fitnesses
        self.pos = np.array(thetas)
        self.fit = np.array(fitnesses)
        
        # Update global best
        best_idx = self.fit.argmin()
        if self.fit[best_idx] < self.gbest_val:
            self.gbest_val = self.fit[best_idx]
            self.gbest_pos = self.pos[best_idx].copy()
        
        lo, hi = self.bounds[:,0], self.bounds[:,1]
        eps = 1e-9
        
        # 1) Employed bees
        for i in range(self.pop_size):
            # choose random partner
            choices = list(range(self.pop_size))
            choices.remove(i)
            k = self.rng.choice(choices)
            d = self.rng.integers(0, self.n_dims)
            phi = self.rng.uniform(-1, 1)
            
            candidate = self.pos[i].copy()
            candidate[d] += phi * (self.pos[i,d] - self.pos[k,d])
            candidate = np.clip(candidate, lo, hi)
            
            cf = self.obj_fn(candidate)
            if cf < self.fit[i]:
                self.pos[i] = candidate
                self.fit[i] = cf
                self.trials[i] = 0
            else:
                self.trials[i] += 1
        
        # 2) Onlooker bees (fitness-proportional selection on inverted costs)
        maxf = self.fit.max()
        inv = maxf - self.fit + eps
        probs = inv / inv.sum()
        
        for _ in range(self.pop_size):
            i = self.rng.choice(self.pop_size, p=probs)
            choices = list(range(self.pop_size))
            choices.remove(i)
            k = self.rng.choice(choices)
            d = self.rng.integers(0, self.n_dims)
            phi = self.rng.uniform(-1, 1)
            
            candidate = self.pos[i].copy()
            candidate[d] += phi * (self.pos[i,d] - self.pos[k,d])
            candidate = np.clip(candidate, lo, hi)
            
            cf = self.obj_fn(candidate)
            if cf < self.fit[i]:
                self.pos[i] = candidate
                self.fit[i] = cf
                self.trials[i] = 0
            else:
                self.trials[i] += 1
        
        # 3) Scout bees (random search for exhausted sources)
        for i in range(self.pop_size):
            if self.trials[i] > self.limit:
                new_sol = self.rng.uniform(lo, hi, size=self.n_dims)
                nf = self.obj_fn(new_sol)
                self.pos[i]    = new_sol
                self.fit[i]    = nf
                self.trials[i] = 0
        
        # Trace population
        self.steps_positions.append(self.pos.tolist())
        self.gen += 1

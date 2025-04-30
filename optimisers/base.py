from abc import ABC, abstractmethod
import numpy as np
rng = np.random.default_rng

class Optimiser(ABC):
    def __init__(self, obj_fn, bounds, seed=0):
        self.obj_fn = obj_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.rng    = rng(seed)
        

    @abstractmethod
    def ask(self):
        ...

    @abstractmethod
    def tell(self, thetas, fitnesses):
        ...

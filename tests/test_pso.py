import unittest
import numpy as np
from optimisers.pso import PSO

def sphere(x):
    return -sum(xi ** 2 for xi in x)  # maximising −x² → minimum at [0,...,0]


def rastrigin(x):
    A = 10
    return -(A * len(x) + sum((xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x))


class TestPSO(unittest.TestCase):
    def setUp(self):
        self.n_dims = 5
        self.bounds = np.array([[-5.0, 5.0]] * self.n_dims)
        self.pop_size = 20
        self.max_gens = 50
        self.seed = 42

    def test_pso_sphere_optimisation(self):
        pso = PSO(sphere, self.bounds, pop_size=self.pop_size, seed=self.seed)
        best_history = []

        for _ in range(self.max_gens):
            thetas = pso.ask()
            scores = [sphere(t) for t in thetas]
            pso.tell(thetas, scores)
            best_history.append(pso.gbest_val)

        # Ensure fitness improves
        self.assertGreater(best_history[-1], best_history[0], "Fitness did not improve.")

        # Check that solution is close to 0
        self.assertGreaterEqual(pso.gbest_val, -1e-2, f"PSO did not converge close to 0: {pso.gbest_val}")

        # Ensure solution is within bounds
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.assertTrue(np.all(pso.gbest_pos >= lo) and np.all(pso.gbest_pos <= hi),
                        "Global best position out of bounds")

    def test_pso_rastrigin_optimisation(self):
        pso = PSO(rastrigin, self.bounds, pop_size=self.pop_size, seed=self.seed)
        best_history = []

        for _ in range(self.max_gens):
            thetas = pso.ask()
            scores = [rastrigin(t) for t in thetas]
            pso.tell(thetas, scores)
            best_history.append(pso.gbest_val)

        # Ensure fitness improves
        self.assertGreater(best_history[-1], best_history[0], "Fitness did not improve on Rastrigin.")

        # Check that result is near the known max (0 for -rastrigin)
        self.assertGreaterEqual(pso.gbest_val, -10.0, f"PSO did not converge close to 0 on Rastrigin: {pso.gbest_val}")
        
if __name__ == "__main__":
    unittest.main(verbosity=2)

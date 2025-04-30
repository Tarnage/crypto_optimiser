import unittest
import numpy as np
from optimisers.gwo import GWO

def sphere(x):
    return sum(xi ** 2 for xi in x)

class TestGWO(unittest.TestCase):
    def setUp(self):
        self.n_dims = 5
        self.bounds = np.array([[-10.0, 10.0]] * self.n_dims)
        self.pop_size = 20
        self.max_gens = 50
        self.seed = 42

    def test_gwo_sphere_optimization(self):
        gwo = GWO(sphere, self.bounds, pop_size=self.pop_size, max_gens=self.max_gens, seed=self.seed)
        best_history = []

        for _ in range(self.max_gens):
            thetas = gwo.ask()
            scores = [sphere(t) for t in thetas]
            gwo.tell(thetas, scores)
            best_history.append(gwo.gbest_val)

        # Test that best fitness decreases
        self.assertLess(best_history[-1], best_history[0], "Fitness did not improve over generations")

        # Test that best fitness is close to 0
        self.assertLessEqual(gwo.gbest_val, 1e-2, f"GWO did not converge close to global minimum: {gwo.gbest_val}")

        # Test that solution is within bounds
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.assertTrue(np.all(gwo.gbest_pos >= lo) and np.all(gwo.gbest_pos <= hi),
                        "Best position is outside of bounds")

        # Test that returned position has correct dimensionality
        self.assertEqual(len(gwo.gbest_pos), self.n_dims)

if __name__ == "__main__":
    unittest.main(verbosity=2)
import numpy as np
import pandas as pd
import time
import itertools
from tqdm.auto import tqdm
from datetime import datetime
from optimisers.abc import ABC
from optimisers.pso import PSO
from optimisers.cab import CAB
from optimisers.gwo import GWO
import json

# --- CONFIGURATION ---
SEEDS = (0, 5)
MAX_GENS = 50
DIM = 6
BOUNDS = np.array([[-5.0, 5.0]] * DIM)
RESULTS_PATH = "results/optimiser_tuning_results/"

# --- FITNESS FUNCTION ---
def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.array(x)
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

# --- PARAMETER GRIDS ---
POP_SIZES = [60, 70, 100]

pso_grid = {
    "pop_size": POP_SIZES,
    "w": [0.8, 0.9],
    "c1": [2.0, 1.8],
    "c2": [2.0, 1.8],
}

cab_grid = {
    "pop_size": POP_SIZES,
    "r_repulsion": [0.3, 0.5],
    "r_orientation": [1.0, 1.2],
    "r_attraction": [1.5, 2.0],
    "alpha": [0.13, 0.16],
    "beta": [0.03, 0.07],
}

gwo_grid = {"pop_size": POP_SIZES}
abc_grid = {"pop_size": POP_SIZES}

# --- UTILITIES ---
def expand_grid(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

# --- WRAPPERS ---
def run_experiment(alg_name, config, seed, tol=1e-12, patience=10):
    start = time.time()

    # Instantiate the correct optimizer
    if alg_name == "PSO":
        opt = PSO(ackley, BOUNDS, seed=seed,
                  pop_size=config["pop_size"], w=config["w"], c1=config["c1"], c2=config["c2"])
    elif alg_name == "CAB":
        opt = CAB(ackley, BOUNDS, seed=seed,
                  pop_size=config["pop_size"],
                  r_repulsion=config["r_repulsion"],
                  r_orientation=config["r_orientation"],
                  r_attraction=config["r_attraction"],
                  alpha=config["alpha"], beta=config["beta"])
    elif alg_name == "GWO":
        opt = GWO(ackley, BOUNDS, seed=seed, max_gens=MAX_GENS,
                  pop_size=config["pop_size"])
    elif alg_name == "ABC":
        opt = ABC(ackley, BOUNDS, seed=seed,
                  pop_size=config["pop_size"])
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    best_val = float("inf")
    stall_count = 0
    gen_count = 0

    for gen in range(MAX_GENS):
        gen_count += 1
        thetas = opt.ask()
        scores = [ackley(t) for t in thetas]
        opt.tell(thetas, scores)

        improvement = best_val - opt.gbest_val
        if improvement < tol:
            stall_count += 1
        else:
            stall_count = 0
            best_val = opt.gbest_val

        if stall_count >= patience:
            break  # Early stop

    return {
        "alg": alg_name,
        "seed": seed,
        "fitness": opt.gbest_val,
        "theta": json.dumps(opt.gbest_pos.tolist()),
        "gens": gen_count,
        "stalled_gens": stall_count,
        "time_sec": round(time.time() - start, 2),
        **config
    }

# --- MAIN ---
if __name__ == "__main__":


    seeds = range(*SEEDS)
    algo_configs = [
        ("PSO", expand_grid(pso_grid)),
        ("CAB", expand_grid(cab_grid)),
        ("GWO", expand_grid(gwo_grid)),
        ("ABC", expand_grid(abc_grid)),
    ]

    for alg, configs in algo_configs:
        algo_results = []
        for config in tqdm(configs, desc=f"{alg} configs"):
            for seed in seeds:
                result = run_experiment(alg, config, seed)
                algo_results.append(result)

        # Save each algorithm’s results to its own CSV
        df = pd.DataFrame(algo_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_PATH}{alg}_results_ackley_hyperparam_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {alg} results to {filename}")


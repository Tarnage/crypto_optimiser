import pandas as pd
import json
import time
from tqdm.auto import tqdm
from optimisers.pso import PSO
from optimisers.gwo import GWO
from optimisers.abc import ABC
from optimisers.cab import CAB
# from optimisers.eho import EHO
# from optimisers.woa import WOA
from core.fitness import evaluate
from core.config import BOUNDS, POP_SIZE, GENS, SEED_ITER, AUX_LOG_FOLDER, TRAIN_TO, DATA_FILE, RESULTS_FOLDER
import core.utils as utils
import multiprocessing as mp
from datetime import datetime
from datetime import timedelta

# -------------  load data  -------------
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
price_all = df["close"].to_numpy()
train_mask = df["date"] <= TRAIN_TO
prices_train = price_all[train_mask]
prices_test  = price_all[~train_mask]


def obj_train(theta, verbose=False):
    """
    Objective function for TRAINING set.
    This is the function that the optimiser will try to minimise.
    """
    profit, aux = evaluate(theta, prices_train)
    if verbose:
        aux_record = {"profit": profit, **aux}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        utils._log_aux(aux_record, theta, f"{AUX_LOG_FOLDER}train_aux_logs_{timestamp}.json")
    return -profit

def obj_test(best_theta, verbose=False):
    """
    Objective function for TEST set.
    This is the function that the optimiser will try to minimise.
    """
    profit, aux = evaluate(best_theta, prices_test)
    if verbose:
        aux_record = {"profit": profit, **aux}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        utils._log_aux(aux_record, best_theta, f"{AUX_LOG_FOLDER}test_aux_logs_{timestamp}.json")
    return -profit


# ------------- Define top level functions(factory) for pickling -------------
def create_pso(seed):
    return PSO(obj_train, BOUNDS, pop_size=POP_SIZE, seed=seed)

def create_cab(seed):
    return CAB(obj_train, BOUNDS,
               pop_size=POP_SIZE, seed=seed,
               r_repulsion=0.312, r_orientation=1.099, r_attraction=1.749,
               alpha=0.144, beta=0.021)

def create_gwo(seed):
    return GWO(obj_train, BOUNDS, pop_size=POP_SIZE, max_gens=GENS, seed=seed)

def create_abc(seed):
    return ABC(obj_train, BOUNDS, pop_size=POP_SIZE, limit=5, seed=seed)

# -------------  experiment loop --------
algs = {
    "PSO": create_pso,
    "CAB": create_cab,
    "GWO": create_gwo,
    "ABC": create_abc,
}

# ------------- parallel worker -------------
def run_single_experiment(name_and_factory_seed, verbose=False, tol=1e-4, patience=10):
    name, factory, run_seed = name_and_factory_seed
    opt = factory(run_seed)
    start = time.time()

    prev_best = float("inf")
    stall_count = 0
    fitness_history = []

    for gen in range(GENS):
        thetas = opt.ask()
        scores = [obj_train(t) for t in thetas]
        opt.tell(thetas, scores)

        current_best = opt.gbest_val
        fitness_history.append(current_best)

        # Check for early stopping
        improvement = prev_best - current_best
        if improvement < tol:
            stall_count += 1
        else:
            stall_count = 0 #reset stall count
            prev_best = current_best

        if stall_count >= patience:
            if verbose:
                print(f"[{name} | seed {run_seed}] Early stopping at generation {gen+1}")
            break

    best_theta = opt.gbest_pos.tolist()
    train_fit  = opt.gbest_val
    test_fit   = obj_test(best_theta, verbose=verbose)
    cleaned_history = [float(x) for x in fitness_history]

    return {
        "alg": name,
        "seed": run_seed,
        "train": train_fit,
        "test": test_fit,
        "theta": json.dumps(best_theta),
        "pop_size": POP_SIZE,
        "gens": gen + 1,
        "stalled_gens": stall_count,
        "tolerance": tol,
        "patience": patience,
        "fitness_history": cleaned_history,
        "time(seconds)": round(time.time() - start, 2),
    }

# ------------- parallel experiment loop -------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # ensures compatibility on all OS
    cpu_count = mp.cpu_count()
    print(f"Using {cpu_count} CPU cores for parallel processing.")
    start_time = time.time() 

    results = []
    all_jobs = []

    seed_start, seed_end = SEED_ITER
    seed_range = range(seed_start, seed_end)
    # Pre-build all jobs for all algorithms
    for name, factory in algs.items():
        for seed in seed_range:
            all_jobs.append((name, factory, seed))

    # Informative outer bar
    outer = tqdm(total=len(all_jobs), desc="Running Experiments", unit="run")

    def callback(res):
        results.append(res)
        outer.update()

    with mp.Pool(processes=min(cpu_count, len(seed_range))) as pool:
        for job in all_jobs:
            pool.apply_async(run_single_experiment, args=(job,), callback=callback)
        pool.close()
        pool.join()

    outer.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(results).to_csv(f"{RESULTS_FOLDER}results_pop{POP_SIZE}_g{GENS}_{timestamp}.csv", index=False)
    total_time = time.time() - start_time
    print(f"\n✅ All experiments completed in {timedelta(seconds=total_time)}")

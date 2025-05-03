import run_experiement as run
import pandas as pd
import time
from datetime import datetime, timedelta
from core.config import POP_SIZE, GENS, RESULTS_FOLDER

#-------------- Define tuple for the experiment --------------
# This is a tuple of the form (name, factory function, seed)
PSO = ("PSO", run.create_pso, 269)
# CAB = ("CAB", run.create_cab, 269)
# GWO = ("GWO", run.create_gwo, 269)
# ABC = ("ABC", run.create_abc, 269)


start_time = time.time()
results = []

run_res = run.run_single_experiment(
    name_and_factory_seed=PSO,
    verbose=True, # Output verbose logs
)

results.append(run_res)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pd.DataFrame(results).to_csv(f"{RESULTS_FOLDER}single_run_results_pop{POP_SIZE}_g{GENS}_{timestamp}.csv", index=False)
total_time = time.time() - start_time
print(f"\n✅ All experiments completed in {timedelta(seconds=total_time)}")
import run_experiement as run
import pandas as pd
import time
from datetime import datetime, timedelta
from core.config import POP_SIZE, GENS, RESULTS_FOLDER

# Best Results for the following parameters:
# alg,seed,train,test,theta,pop,gens
# ABC,3700,-62523.399027568434,-9090.518801394715,"[6.017094713469811, 1.949066275064014, 0.05025449671474211, 62.62690987559359, 1.9041361355438666, 0.37889543781706464, 6.979556788661506, 10.00627544404015]",50,100
# CAB,2891,-47539.07266848066,-9809.076537755618,"[3.7496667813184064, 0.9910610154449132, 0.3538166221746755, 172.59853444440796, 2.0189255408579316, 0.4608990795658156, 11.727296386716285, 9.963769150571828]",50,100
# GWO,3695,-61074.16272729421,-9148.248267688092,"[6.1318525243658515, 0.0, 0.05, 171.7304152782868, 2.039099407866258, 0.3711214440167874, 7.098675754276093, 9.810925455925856]",50,100
# PSO,269,-56656.63745214293,-8684.058276382995,"[5.507079479923081, 0.26241909154185505, 0.4014952617968686, 169.4378223337535, 2.092276298354577, 0.3823459238469119, 6.977768626738198, 9.0]",50,100


#-------------- Define tuple for the experiment --------------
# This is a tuple of the form (name, factory function, seed)
# PSO = ("PSO", run.create_pso, 269)
# CAB = ("CAB", run.create_cab, 2891)
GWO = ("GWO", run.create_gwo, 3695)
# ABC = ("ABC", run.create_abc, 3700)


start_time = time.time()
results = []

run_res = run.run_single_experiment(
    name_and_factory_seed=GWO,
    verbose=True, # Output verbose logs
)

results.append(run_res)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pd.DataFrame(results).to_csv(f"{RESULTS_FOLDER}single_run_results_pop{POP_SIZE}_g{GENS}_{timestamp}.csv", index=False)
total_time = time.time() - start_time
print(f"\n✅ All experiments completed in {timedelta(seconds=total_time)}")
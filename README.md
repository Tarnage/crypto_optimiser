# 📈 Trading Bot README

This project implements a **backtestable, evaluation-ready crossover trading bot** for cryptocurrency markets (tested on BTC daily data).  
The bot logic and fitness evaluation, are fully modular and optimised for testing with nature-inspired optimisation algorithms (CAB, PSO, GWO, ABC).
It supports full experimentation, single-run evaluation, and visual analysis using historical BTC-Daily data.

## 🚀 Features

- Dual moving-average crossover strategy (configurable)

- 4 optimizers supported: PSO, CAB, GWO, ABC

- Multi-core experiment execution

- 3% trading fee built into simulation

- Detailed backtest metrics: drawdown, win rate, trade return, and more

- Configurable via core/config.py

- Produces test/train fitness and exportable CSVs


## 🗂️ Folder Structure
```
.
├── data/                  # Historical BTC data (CSV)
├── results/               # Experiment outputs
├── logs/                  # Optional detailed logs
├── core/                  # Signal logic, evaluation, config
├── optimisers/            # PSO, CAB, ABC, GWO implementations
├── run_experiment.py      # Batch experiment script
├── run_single.py          # Run one optimizer with one seed
├── forward_test.ipynb     # Visualize crossover results
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 📥 Setup

1. Clone this repo

2. Install dependencies via `requirements.txt`

3. Put your BTC data in `data/BTC-Daily.csv`

## ⚙️ Configuration

All key settings are located in `core/config.py`

```python
BOUNDS = [
  (2, 10),      # d1: Fast MA window
  (0, 3),       # t1: MA type (0=SMA, 1=LMA, 2=EMA, 3=MACD)
  (0.05, 0.5),  # a1: Smoothing alpha

  (20, 400),    # d2: Slow MA window
  (0, 3),       # t2: MA type
  (0.05, 0.5),  # a2: Smoothing alpha
  (5.0, 12.0),  # buy_delay
  (5.0, 15.0),  # sell_delay
]

POP_SIZE = 50
GENS = 100                 # Maximum number of generation/epochs iterations.

# SEED_ITER = (start, end): run each optimizer from start to end-1
# e.g. (2001, 4000) will perform 1999 runs per algorithm
SEED_ITER = (2001, 4000)  

DATA_FILE = "data/BTC-Daily.csv"
TRAIN_TO = "2019-12-31"
RESULTS_FOLDER = "results/"
AUX_LOG_FOLDER = "logs/"
```

## 📊 How to Run

### 🔁 Run Batch Experiments

Run all optimizers across multiple seeds:

```python
python run_experiment.py
```

This saves CSV files to results/ with detailed metrics and configurations.

### 🔎 Run a Single Experiment

Manually test a specific optimizer with a seed:

```python
python run_single_experiment.py
```

This is great for debugging or visual analysis.

## 📤 Output Format

Example output in CSV:

```python
alg,seed,train,test,theta,pop_size,gens,...
PSO,9,-1497.65,-486.12,"[5.2, 1, 0.1, 100, 2, 0.2, 5, 10]",50,12,...
```

Includes:
- train, test fitness
- theta: trading strategy parameters
- fitness_history: progress over generations
- more information can be found in `results\header_description.md`

## ✅ Example Visual
Can be found `foward_test.ipynb`

## 🔧 How to Add a New Optimizer

1. Create a new Python file in `optimisers/` (e.g. `my_algo.py`)

2. Implement a class following the base structure (`optimisers/base.py`):

```python
class Optimiser:
    def __init__(self, obj_fn, bounds, seed=0):
        ...
    def ask(self):
        ...
    def tell(self, thetas, scores):
        ...
```

3. Add a factory function to run_experiment.py:

```python
def create_my_algo(seed):
    return MyAlgo(obj_train, BOUNDS, pop_size=POP_SIZE, max_gens=GENS, seed=seed)
```

4. Register your algorithm in the algs dictionary:

```python
algs = {
    "PSO": create_pso,
    "CAB": create_cab,
    "MY_ALGO": create_my_algo
}
```
FAST_WINDOWS = [5, 10, 20, 50] # not used
SLOW_WINDOWS = [20, 50, 100, 200] # not used

# -------------  bounds  ----------------
# BOUNDS = [
#   (0, len(FAST_WINDOWS) - 1),   # d₁ = fast-index ∈ {0,1,2,3}
#   (0,    2),    # t₁ ∈ [0,2]               (fast‐MA type code: 0=SMA,1=LMA,2=EMA)
#   (0.05, 0.5),  # α₁ ∈ [0.05,0.5]          (fast‐EMA decay; only used if t₁==2)

#   (0, len(SLOW_WINDOWS) - 1),  # d₂ = slow-index ∈ {0,1,2,3}
#   (0,    2),    # t₂ ∈ [0,2]               (slow‐MA type code)
#   (0.05, 0.5)   # α₂ ∈ [0.05,0.5]          (slow‐EMA decay; only if t₂==2)
# ]

# run_experiment.py
# Define hyperparameters for the optimisers
BOUNDS = [
  (2, 30),      # d₁ = fast window 
  (0,    3),    # t₁ ∈ [0,2]               (fast‐MA type code: 0=SMA,1=LMA,2=EMA,3=MACD)
  (0.05, 0.5),  # α₁ ∈ [0.05,0.5]          (fast‐EMA decay; only used if t₁==2)

  (20, 200),    # d₂ = slow window
  (0,    3),    # t₂ ∈ [0,2]               (slow‐MA type code)
  (0.05, 0.5),   # α₂ ∈ [0.05,0.5]          (slow‐EMA decay; only if t₂==2)
  (3.0, 9.0),   # shit = shift/lag (for buy delay)
  (2.0, 9.0),   # shit = shift/lag (for sell delay)
]
POP_SIZE = 50          # population size
GENS = 100                   # number of generations
LAMBDA = 5.0          # try 1, 5, 10 … and see what moves the gap histogram
SEED_ITER = 20         # number of runs per algorithm

# Define log files

AUX_LOG_FOLDER = "logs/"

# Define the path to the data file
TRAIN_TO = "2019-12-31" # date to split train/test data
DATA_FILE = "data/BTC-Daily.csv" # path to the data file
# DATA_FILE = "data/BTC-Hourly.csv" 
# Define results file
RESULTS_FOLDER = "results/" # path to the results file

# core.bot.py
# Define the number of candles to shift the signal by (for crossover detection)
# TODO: could be a bounds parameter
SHIFT = 4

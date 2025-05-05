# Bot hyperparameters. 
# Window sizes and indicator types
BOUNDS = [
  (2, 10),      # fast window size (d1)
  (0, 3),       # fast MA type: 0=SMA, 1=LMA, 2=EMA, 3=MACD (t1)
  (0.05, 0.5),  # fast smoothing (alpha)

  (20, 400),    # slow window size (d2)
  (0, 3),       # slow MA type: 0=SMA, 1=LMA, 2=EMA, 3=MACD (t2)
  (0.05, 0.5),  # slow smoothing (alpha)
  (5.0, 12.0),  # buy delay (lag)
  (5.0, 15.0),  # sell delay (lag)
]

# Optimizer parameters
POP_SIZE = 50
GENS = 100
SEED_ITER = (0, 20)    # Seed range for repeatability

# File paths
DATA_FILE = "data/BTC-Daily.csv"
TRAIN_TO = "2019-12-31"
RESULTS_FOLDER = "results/"
AUX_LOG_FOLDER = "logs/"
import json
import os
from datetime import datetime

def _log_aux(aux, theta, log_file):
    """
    Append a JSON line with timestamp, theta, and aux metrics.
    """

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "theta": theta,
        "metrics": aux
    }
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")


import json
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, Dict


def save_pickle(path: Path, ob: Any):
    with open(path, "wb") as f:
        pkl.dump(ob, f)


def save_json(path: Path, ob: Dict):
    with open(path, "w") as f:
        json.dump(ob, f, indent=4)


def generate_plots(path: Path, metrics_history: Dict):
    path.mkdir(parents=True)
    for metric, content in metrics_history.items():
        sns.lineplot(content)
        plt.title(metric)
        plt.savefig(path / f"{metric}.jpg")
        plt.clf()

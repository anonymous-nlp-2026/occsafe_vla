import json
import numpy as np

TASK_SHORT = {i: f"T{i}" for i in range(10)}

_cache = {}


def load_eval_json(path):
    if path in _cache:
        return _cache[path]
    with open(path) as f:
        data = json.load(f)
    result = {}
    for task_name, task_data in data["tasks"].items():
        tid = task_data["task_id"]
        eps = task_data["episodes"]
        result[tid] = {
            "task_name": task_name,
            "tsr": task_data["tsr"],
            "car": task_data["car"],
            "n": task_data["n"],
            "success_binary": [int(e["success"]) for e in eps],
            "safe_binary": [1 - int(e["collision"]) for e in eps],
        }
    _cache[path] = result
    return result


def pool_binary(json_paths, task_id, metric="safe_binary"):
    pooled = []
    for p in json_paths:
        data = load_eval_json(p)
        if task_id in data:
            pooled.extend(data[task_id][metric])
    return np.array(pooled, dtype=float)


def seed_level_metric(json_paths, task_id, metric="car"):
    vals = []
    for p in json_paths:
        data = load_eval_json(p)
        if task_id in data:
            vals.append(data[task_id][metric])
    return np.array(vals)


def bootstrap_ci(arr, n_bootstrap=10000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boots = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def bootstrap_diff_test(arr1, arr2, n_bootstrap=10000):
    arr1, arr2 = np.asarray(arr1, dtype=float), np.asarray(arr2, dtype=float)
    if len(arr1) == 0 or len(arr2) == 0:
        return np.nan, np.nan, np.nan, np.nan
    obs = float(arr1.mean() - arr2.mean())
    rng = np.random.default_rng(42)
    idx1 = rng.integers(0, len(arr1), size=(n_bootstrap, len(arr1)))
    idx2 = rng.integers(0, len(arr2), size=(n_bootstrap, len(arr2)))
    diffs = arr1[idx1].mean(axis=1) - arr2[idx2].mean(axis=1)
    p = float(np.mean(np.abs(diffs) >= np.abs(obs)))
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), p


def get_task_ids(json_paths):
    for p in json_paths:
        data = load_eval_json(p)
        return sorted(data.keys())
    return list(range(10))


def parse_condition(s):
    label, paths_str = s.split(":", 1)
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    return label, paths


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt

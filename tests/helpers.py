"""SYNTHETIC fixture data for tests only (not real housing data)."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_california_like(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "MedInc": rng.uniform(1, 10, n),
        "HouseAge": rng.uniform(1, 52, n),
        "AveRooms": rng.uniform(3, 8, n),
        "AveBedrms": rng.uniform(1, 1.5, n),
        "Population": rng.uniform(300, 3000, n),
        "AveOccup": rng.uniform(2, 4, n),
        "Latitude": rng.uniform(32.5, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
    })
    df["price_usd"] = 50_000 + 40_000 * df["MedInc"] + 500 * df["HouseAge"] + rng.normal(0, 15_000, n)
    return df


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path

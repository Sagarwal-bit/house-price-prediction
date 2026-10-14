# Data

No data file is committed.

## Default dataset: California Housing

- Loaded by `python -m src.train` through `sklearn.datasets.fetch_california_housing` (downloads once, cached in `~/scikit_learn_data`; internet needed the first time).
- Origin: 1990 US census, one row per **census block group** (not per house); 20,640 rows per the scikit-learn documentation. Pace & Barry (1997), distributed through StatLib.
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` (all numeric).
- Target: `MedHouseVal` (median house value in units of $100,000), converted to `price_usd`. Values are capped near $500,000. Prices are **1990 US dollars**.
- Licence: I could not verify a formal licence text from the build environment. Check the scikit-learn dataset documentation and the StatLib source before redistributing the data; this repository does not redistribute it.
- Actual row counts after validation (duplicates, invalid values) are measured by training and stored in `models/metrics.json`.

## If the download fails

- Retry with a working internet connection, or
- supply your own CSV with the same columns plus `price_usd` (or `MedHouseVal`): `python -m src.train --data path/to/file.csv`, or
- use the clearly-labelled **demo** fallback (the original internship formula-generated data): `python -m src.train --dataset synthetic`. A model trained this way is a demonstration only.

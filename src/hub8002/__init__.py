"""hub8002: exposure- and position-aware hub detection in football passing networks."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    cfg = yaml.safe_load(open(path or ROOT / 'config.yaml'))
    cfg['seasons'] = {int(k): v for k, v in cfg['seasons'].items()}
    return cfg


def save_table(df, name, folder=None):
    """Parquet when pyarrow is available, else pickle (same stem)."""
    folder = folder or ROOT / 'data' / 'interim'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(folder / f'{name}.parquet', index=False)
    except ImportError:
        df.to_pickle(folder / f'{name}.pkl')


def load_table(name, folder=None):
    import pandas as pd
    folder = folder or ROOT / 'data' / 'interim'
    pq = folder / f'{name}.parquet'
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(folder / f'{name}.pkl')

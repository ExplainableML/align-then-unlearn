from pathlib import Path


def get_data_root():
    return Path(__file__).parent.parent.parent / "data/rwku/benchmark/Target"

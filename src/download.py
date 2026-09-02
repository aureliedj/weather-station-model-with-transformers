"""Download the PeakWeather dataset.

    python src/download.py                       # -> <repo>/PeakWeatherDataset
    DATA_ROOT=/somewhere python src/download.py
"""
import os

from peakweather.dataset import PeakWeatherDataset

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "PeakWeatherDataset"),
)

if __name__ == "__main__":
    print(f"Downloading PeakWeather to {DATA_ROOT}")
    PeakWeatherDataset(root=DATA_ROOT)
    print(f"Done. Use it with:  export DATA_ROOT={DATA_ROOT}")

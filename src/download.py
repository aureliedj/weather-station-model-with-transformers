"""Fetch the PeakWeather dataset used by this project.

Usage:
    python src/download.py                    # -> ./PeakWeatherDataset
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
    print(f"Downloading PeakWeather to: {DATA_ROOT}")
    PeakWeatherDataset(root=DATA_ROOT)
    print("Done. Point the launchers at it with:  export DATA_ROOT=" + DATA_ROOT)

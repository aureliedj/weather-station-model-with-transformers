from peakweather.dataset import PeakWeatherDataset

path = "/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset"

ds = PeakWeatherDataset(root = path)

# Show dataset information
print(f"Number of time steps: {ds.num_time_steps}")


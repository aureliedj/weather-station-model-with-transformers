from peakweather.dataset import PeakWeatherDataset
ds = PeakWeatherDataset(root="/Users/aureliedejong/Documents/ETH/_DAS Project/PeakWeatherDataset")

print(ds.get_observations(stations='KLO'))
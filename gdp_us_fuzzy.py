import pandas as pd
from utils.analyzer import DataAnalyzer
from utils.visualizer import DataVisualizer
from pyFTS.common import Transformations, Util, Membership
from pyFTS.partitioners import Grid, partitioner
from pyFTS.models import chen, cheng, hofts, pwfts
import matplotlib.pyplot as plt
from main.utils.common import theil
from sklearn.metrics import mean_absolute_percentage_error, r2_score

analyzer = DataAnalyzer(file_name="gdp")
visualizer = DataVisualizer()

# reading time series data from csv file and prepare for analysis
data = analyzer.load_and_prepare_data()

train_size = int(0.8 * len(data))
train = data[:train_size]
test = data[train_size-2:]

tdiff = Transformations.Differential(1)
train = tdiff.apply(data)

fs = Grid.GridPartitioner(data=train, npart=5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25,10])
fs.plot(ax)

''' uncomment to try other models '''
# model = chen.ConventionalFTS(partitioner=fs)
model = pwfts.ProbabilisticWeightedFTS(partitioner=fs)
# model = cheng.TrendWeightedFTS(partitioner=fs)

model.fit(train)
model.append_transformation(tdiff)

preds = model.predict(train)
fig, ax = plt.subplots(figsize=(20,10))

ax = pd.DataFrame(train).plot(figsize=(15, 5))
pd.DataFrame(preds).plot(ax=ax)
plt.legend(['True Data', 'Forecasts'])
ax.set_title('Raw Data and Forecasts on the train set')
plt.show()

print(model)

Util.plot_rules(model, size=[15, 10], rules_by_axis=10)

forecasts = model.predict(test)
data = pd.DataFrame(test)
forecasts = pd.DataFrame(forecasts)

mape = mean_absolute_percentage_error(test, forecasts)
print("MAPE:")
print(mape)
print("Theil:")
print(theil(test, forecasts))

r2 = r2_score(test, forecasts)
print(r2)

fig, ax = plt.subplots(figsize=(20,10))

ax = data.plot(figsize=(15, 5))
forecasts.plot(ax=ax)
plt.legend(['True Data', 'Forecasts'])
ax.set_title('Raw Data and Forecasts on the test set')
plt.show()
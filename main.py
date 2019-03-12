from data import Data
from selfmonitoring import SelfMonitoring

# Simulates data generation from several units at each day ("1d")
dataset = Data("../data/", freq="1d")

# Create an instance of SelfMonitoring
sm = SelfMonitoring(rep_type="hist",           # Data representation type: "stats" or "hist"
                    w_ref_group="7days",        # Time window for the reference group
                    w_martingale=15,            # Window size for computing the deviation level
                    non_conformity="median",    # Non-conformity (strangeness) measure: "median" or "knn"
                    k=50,                       # Used if non_conformity is "knn"
                    dev_threshold=.6)           # Threshold on the deviation level

# At each day (dt), data is generated from several units (dfs)
# Each dataframe df in dfs, comes from one unit over the day dt
for dt, dfs in dataset.getNext():
    # diagnoise the selected test unit (at index 0)
    strangeness, pvalue, deviation, is_dev = sm.diagnoise(0, dt, dfs)
    
    print("Time: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})"
        .format(dt, strangeness, pvalue, deviation, "high" if is_dev else "low"))

# Plot p-values and deviation level over time
sm.plot_deviation()

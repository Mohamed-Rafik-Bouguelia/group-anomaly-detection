from cosmo.datasets import load_vehicles
from cosmo import GroupAnomaly

if __name__ == '__main__':
    
    # Streams data from several units (vehicles) over time
    dataset = load_vehicles()
    nb_units = dataset.get_nb_units() # 19 vehicles in this example

    # Create an instance of GroupAnomaly
    gdev = GroupAnomaly(  nb_units=nb_units,          # Number of units (vehicles)
                            ids_target_units=[0, 1],    # Ids of the (target) units to diagnoise
                            w_ref_group="7days",        # Time window for the reference group
                            w_martingale=15,            # Window size for computing the deviation level
                            non_conformity="lof",    # Non-conformity (strangeness) measure: "median" or "knn" or "lof"
                            k=50,                       # Used if non_conformity is "knn"
                            dev_threshold=.6)           # Threshold on the deviation level

    # At each time dt, x_units contains data from all units.
    # Each data-point x_units[i] comes from the i'th unit.
    for dt, x_units in dataset.stream():
        
        # diagnoise the selected target units (0 and 1)
        devContextList = gdev.predict(dt, x_units)
        
        for uid, devCon in enumerate(devContextList):
            print("Unit:{}, Time: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})".format(uid, dt, devCon.strangeness, 
            devCon.pvalue, devCon.deviation, "high" if devCon.is_deviating else "low"))

    # Plot p-values and deviation levels over time
    gdev.plot_deviations()

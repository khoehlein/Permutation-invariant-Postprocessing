import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.prediction_directory import PredictionStorage


def main():
    dataset = 'test'
    flt = 24
    store_ref = PredictionStorage.from_model_name('DRN', flt)
    reference = store_ref.sample_ensemble(dataset)
    observations = store_ref.get_observations(dataset)
    pit_ref = reference.compute_pit(observations)
    crps_ref = reference.compute_crps(observations)
    bins_ref = np.floor(pit_ref * 20).astype(int)

    model_name = 'ED-DRN'
    store = PredictionStorage.from_model_name(model_name, flt)
    prediction = store.sample_ensemble(dataset)
    pit = prediction.compute_pit(observations)
    crps = prediction.compute_crps(observations)
    bins = np.floor(pit * 20).astype(int)

    df = pd.DataFrame({'bin': bins, 'bin_ref': bins_ref, 'crps': crps, 'crps_ref': crps_ref})
    crps_mean = df.groupby(by=['bin', 'bin_ref']).mean().to_xarray()


    fig, ax = plt.subplots(1, 1)
    crpss = 1. - crps_mean['crps'] / crps_mean['crps_ref']
    vmax = np.max(np.abs(crpss.values))
    p = ax.pcolor(crpss.values, vmin=-vmax, vmax=vmax, cmap='coolwarm')
    plt.colorbar(p, ax=ax)
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(pit, pit_ref, alpha=0.1)
    plt.tight_layout()
    plt.show()
    plt.close()




if __name__ == '__main__':
    main()

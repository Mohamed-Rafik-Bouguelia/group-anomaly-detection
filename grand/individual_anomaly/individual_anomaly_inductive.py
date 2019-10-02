"""An inductive anomaly detection model for an individual system.
To detect anomalies, the IndividualAnomalyInductive model compares the data from
a system against a fixed reference dataset.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from grand.conformal import get_strangeness
from grand.utils import DeviationContext
from grand import utils, plotting
import matplotlib.pylab as plt
import pandas as pd
import numpy as np


class IndividualAnomalyInductive:
    """Deviation detection for a single/individual unit

    Parameters:
    ----------
    w_martingale : int
        Window used to compute the deviation level based on the last w_martingale samples.

    non_conformity : string
        Strangeness (or non-conformity) measure used to compute the deviation level.
        It must be either "median" or "knn"

    k : int
        Parameter used for k-nearest neighbours, when non_conformity is set to "knn"

    dev_threshold : float
        Threshold in [0,1] on the deviation level
    """
    
    def __init__(self, w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6, columns=None):
        utils.validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold)
        
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        self.columns = columns
        
        self.strg = get_strangeness(non_conformity, k)
        self.T, self.S, self.P, self.M = [], [], [], []
        self.representatives, self.diffs = [], []
        
        self.mart = 0
        self.marts = [0, 0, 0]

        self.history_data = []

    # ===========================================
    def fit(self, X):
        '''Fit the anomaly detector to the data X (assumed to be normal)
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples assumed to be not deviating from normal
        
        Returns:
        --------
        self : object
        '''
        
        self.strg.fit(X)
        return self
    
    # ===========================================
    def predict(self, dtime, x):
        '''Update the deviation level based on the new test sample x
        
        Parameters:
        -----------
        dtime : datetime
            datetime corresponding to the sample x
        
        x : array-like, shape (n_features,)
            Sample for which the strangeness, p-value and deviation level are computed
        
        Returns:
        --------
        strangeness : float
            Strangeness of x with respect to samples in Xref
        
        pval : float, in [0, 1]
            p-value that represents the proportion of samples in Xref that are stranger than x.
        
        deviation : float, in [0, 1]
            Normalized deviation level updated based on the last w_martingale steps
        '''
        
        self.T.append(dtime)
        self.history_data.append(x)

        strangeness, diff, representative = self.strg.predict(x)
        self.S.append(strangeness)
        self.diffs.append(diff)
        self.representatives.append(representative)

        pval = self.strg.pvalue(strangeness)
        self.P.append(pval)
        
        deviation = self._update_martingale(pval)
        self.M.append(deviation)
        
        is_deviating = deviation > self.dev_threshold # TODO: remove from deviation context in future
        return DeviationContext(strangeness, pval, deviation, is_deviating)
        
    # ===========================================
    def _update_martingale(self, pval):
        '''Incremental additive martingale over the last w_martingale steps.
        
        Parameters:
        -----------
        pval : int, in [0, 1]
            The most recent p-value
        
        Returns:
        --------
        normalized_one_sided_mart : float, in [0, 1]
            Deviation level. A normalized version of the current martingale value.
        '''
        
        betting = lambda p: -p + .5
        
        self.mart += betting(pval)
        self.marts.append(self.mart)
        
        w = min(self.w_martingale, len(self.marts))
        mat_in_window = self.mart - self.marts[-w]
        
        normalized_mart = ( mat_in_window ) / (.5 * w)
        normalized_one_sided_mart = max(normalized_mart, 0)
        return normalized_one_sided_mart
        
    # ===========================================
    def get_stats(self):
        stats = np.array([self.S, self.P, self.M]).T
        return pd.DataFrame(index=self.T, data=stats, columns=["strangeness", "pvalue", "deviation"])

    # ===========================================
    def get_all_deviations(self, min_len=5, dev_threshold=None):
        if dev_threshold is None: dev_threshold = self.dev_threshold

        arr = np.arange(len(self.M))
        boo = np.array(self.M) > dev_threshold
        indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
        groups_ids = np.split(arr, indices)
        groups_ids = groups_ids[0::2] if boo[0] else groups_ids[1::2]

        diffs_df = pd.DataFrame(index=self.T, data=self.diffs)
        sub_diffs_dfs = [diffs_df.iloc[ids, :] for ids in groups_ids if len(ids) >= min_len]

        dev_signatures = [np.mean(sub_diffs_df.values, axis=0) for sub_diffs_df in sub_diffs_dfs]
        periods = [(sub_diffs_df.index[0], sub_diffs_df.index[-1]) for sub_diffs_df in sub_diffs_dfs]

        deviations = [(devsig, p_from, p_to) for (devsig, (p_from, p_to)) in zip(dev_signatures, periods)]
        return deviations

    # ===========================================
    def get_deviation_signature(self, from_time, to_time):
        sub_diffs_df = pd.DataFrame(index=self.T, data=self.diffs)[from_time: to_time]
        deviation_signature = np.mean(sub_diffs_df.values, axis=0)
        return deviation_signature

    # ===========================================
    def get_similar_deviations(self, from_time, to_time, k_devs=2, min_len=5, dev_threshold=None):
        target_devsig = self.get_deviation_signature(from_time, to_time)
        deviations = self.get_all_deviations(min_len, dev_threshold)
        dists = [np.linalg.norm(target_devsig - devsig) for (devsig, *_) in deviations]
        ids = np.argsort(dists)[:k_devs]
        return [deviations[id] for id in ids]

    # ===========================================
    def plot_deviations(self, figsize=None, savefig=None, plots=None):
        df_data = pd.DataFrame(index=self.T, data=self.history_data, columns=self.columns)
        if self.columns is None: df_data = df_data.add_prefix('Feature ')
        df_representatives = pd.DataFrame(index=self.T, data=self.representatives, columns=self.columns).add_prefix('Representatives ')
        df_dev_contexts = self.get_stats()

        plotting.plot_deviations(df_data, df_representatives, df_dev_contexts, self.dev_threshold, figsize, savefig, plots)

    # ===========================================
    def plot_explanations(self, from_time, to_time, figsize=None, savefig=None, k_features=4):
        # TODO: validate if the period (from_time, to_time) has data before plotting

        from_time_pad = from_time - (to_time - from_time)
        to_time_pad = to_time + (to_time - from_time)

        df = pd.DataFrame(index=self.T, data=self.history_data)
        sub_df = df[from_time: to_time]
        sub_df_before = df[from_time_pad: from_time]
        sub_df_after = df[to_time: to_time_pad]

        sub_representatives_df_pad = pd.DataFrame(index=self.T, data=self.representatives)[from_time_pad: to_time_pad]
        sub_diffs_df = pd.DataFrame(index=self.T, data=self.diffs)[from_time: to_time]

        nb_features = sub_diffs_df.values.shape[1]
        if (self.columns is None) or (len(self.columns) != nb_features):
            self.columns = ["Feature {}".format(j) for j in range(nb_features)]
        self.columns = np.array(self.columns)

        features_scores = np.array([np.abs(col).mean() for col in sub_diffs_df.values.T])
        features_scores = 100 * features_scores / features_scores.sum()
        k_features = min(k_features, nb_features)
        selected_features_ids = np.argsort(features_scores)[-k_features:][::-1]
        selected_features_names = self.columns[selected_features_ids]
        selected_features_scores = features_scores[selected_features_ids]

        fig, axs = plt.subplots(k_features, figsize=figsize)
        fig.autofmt_xdate()
        fig.suptitle("Ranked features\nFrom {} to {}".format(from_time, to_time))
        if k_features == 1 or not isinstance(axs, (np.ndarray) ):
            axs = np.array([axs])

        for i, (j, name, score) in enumerate(zip(selected_features_ids, selected_features_names, selected_features_scores)):
            print("{0}, score: {1:.2f}%".format(name, score))
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("{0}\n(Score: {1:.1f})".format(name, score))
            axs[i].plot(sub_representatives_df_pad.index, sub_representatives_df_pad.values[:, j], color="grey", linestyle='--')
            axs[i].plot(sub_df_before.index, sub_df_before.values[:, j], color="green")
            axs[i].plot(sub_df_after.index, sub_df_after.values[:, j], color="lime")
            axs[i].plot(sub_df.index, sub_df.values[:, j], color="red")

        figg = None
        if k_features > 1:
            figg, ax1 = plt.subplots(figsize=figsize)
            figg.suptitle("Top 2 ranked features\nFrom {} to {}".format(from_time, to_time))
            (j1, j2) , (nm1, nm2), (s1, s2) = selected_features_ids[:2], selected_features_names[:2], selected_features_scores[:2]

            ax1.set_xlabel("{0}\n(Score: {1:.1f})".format(nm1, s1))
            ax1.set_ylabel("{0}\n(Score: {1:.1f})".format(nm2, s2))

            self.strg.X = np.array(self.strg.X)
            ax1.scatter(self.strg.X[:, j1], self.strg.X[:, j2], color="silver", marker=".", label="Reference Data")
            ax1.scatter(sub_df_before.values[:, j1], sub_df_before.values[:, j2], color="green", marker=".", label="Before Selected Period")
            ax1.scatter(sub_df_after.values[:, j1], sub_df_after.values[:, j2], color="lime", marker=".", label="After Selected Period")
            ax1.scatter(sub_df.values[:, j1], sub_df.values[:, j2], color="red", marker=".", label="Selected Period")
            ax1.legend()

        if savefig is None:
            plt.show()
        else:
            figpathname = utils.create_directory_from_path(savefig)
            fig.savefig(figpathname)
            if figg is not None: figg.savefig(figpathname + "_2.png")

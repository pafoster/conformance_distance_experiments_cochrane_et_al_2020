import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

def compute_roc_curve(variance_inliers, variance_outliers):
    scores = np.append(variance_inliers, variance_outliers)
    class_assignments = np.append(np.zeros(len(variance_inliers)),
                                  np.ones(len(variance_outliers))).astype(bool)

    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(class_assignments, scores)
    area_under_curve = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    
    return false_positive_rate, true_positive_rate, area_under_curve

def plot_roc_curve(variance_inliers, variance_outliers):
    plt.figure(figsize=(6, 6))
    for key in variance_inliers.keys():
        false_positive_rate, true_positive_rate, area_under_curve = compute_roc_curve(variance_inliers[key],
                                                                                      variance_outliers[key])
        plt.plot(false_positive_rate, true_positive_rate,
                 linewidth=3, label='{} (AUC={:0.3f})'.format(key, area_under_curve))
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5), borderaxespad=0)
    plt.grid(True)
    plt.show()

def bootstrap_estimate_standard_error(variance_inliers, variance_outliers, n_samples=10**4):
    n = len(variance_inliers)
    auc_values = np.empty(n_samples)
    for i in range(n_samples):
        I = np.random.choice(n, n, replace=True)
        *_, area_under_curve = compute_roc_curve(variance_inliers[I], variance_outliers[I])
        auc_values[i] = area_under_curve
    
    return np.std(auc_values, ddof=1)

def tabulate_performance(variance_inliers_vs_hyperparams, variance_outliers_vs_hyperparams):
    auc_vs_hyperparameters = {}
    standard_error_vs_hyperparameters = {}
    for hyperparam in variance_inliers_vs_hyperparams:
        auc_vs_hyperparameters[hyperparam] = {}
        standard_error_vs_hyperparameters[hyperparam] = {}

        for method in variance_inliers_vs_hyperparams[hyperparam]:
            np.random.seed(5)
            *_, auc = compute_roc_curve(variance_inliers_vs_hyperparams[hyperparam][method], variance_outliers_vs_hyperparams[hyperparam][method])
            standard_error = bootstrap_estimate_standard_error(variance_inliers_vs_hyperparams[hyperparam][method],
                                                               variance_outliers_vs_hyperparams[hyperparam][method])
            auc_vs_hyperparameters[hyperparam][method] = auc
            standard_error_vs_hyperparameters[hyperparam][method] = standard_error

    auc_vs_hyperparameters = pd.DataFrame(auc_vs_hyperparameters).applymap(lambda value: '{:0.3f}'.format(value))
    standard_error_vs_hyperparameters = pd.DataFrame(standard_error_vs_hyperparameters).applymap(lambda value: '{:0.3f}'.format(value))

    return auc_vs_hyperparameters, standard_error_vs_hyperparameters

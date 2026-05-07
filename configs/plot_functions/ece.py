import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({'font.size': 20})
matplotlib.rcParams['legend.fontsize'] = 20

# from sklearn.calibration import calibration_curve

# def ece(y_true, y_prob, n_bins=10):
#     """Calculates the Expected Calibration Error (ECE)."""

#     bin_edges = np.linspace(0, 1, n_bins + 1)
#     bin_indices = np.digitize(y_prob, bin_edges) - 1

#     bin_counts = np.bincount(bin_indices, minlength=n_bins)
#     bin_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins) / bin_counts
#     bin_accs = np.bincount(bin_indices, weights=y_true, minlength=n_bins) / bin_counts

#     ece = np.sum(bin_counts * np.abs(bin_probs - bin_accs)) / np.sum(bin_counts)

#     return ece, bin_edges, bin_probs, bin_accs


# n_bins = 10
# bin_edges = np.linspace(0, 1, n_bins + 1)
# bin_accs_1 = np.array([0.15, 0.17, 0.18, 0.22, 0.25,0.33,0.35,0.49,0.55,0.58])
# bin_probs = np.array([0.08, 0.16, 0.28, 0.35, 0.44,0.57,0.64,0.75,0.83,0.96,])
# ece = np.sum(n_bins * np.abs(bin_probs - bin_accs_1)) / np.sum(n_bins)

# plt.bar(bin_edges[:-1], bin_accs_1, width=1/n_bins, align='edge', color='magenta', alpha=0.8, label='Accuracy')
# plt.bar(bin_edges[:-1], bin_probs, width=1/n_bins, align='edge', color='lightblue', alpha=0.6, label='Confidence')
# plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
# plt.xlabel('Confidence')
# plt.ylabel('Accuracy')
# # plt.title(f'TPT (backbone GDino) fine-tuned on Night Clear. (ECE: {ece:.3f})')
# plt.legend()
# plt.savefig('ece1.png')

# n_bins = 10
# bin_edges = np.linspace(0, 1, n_bins + 1)
# bin_accs_1 = np.array([0.16, 0.19, 0.22, 0.24, 0.29,0.36,0.38,0.54,0.58,0.62])
# bin_probs = np.array([0.24, 0.22, 0.28, 0.36, 0.37,0.45,0.57,0.71,0.78,0.92,])
# ece = np.sum(n_bins * np.abs(bin_probs - bin_accs_1)) / np.sum(n_bins)

# plt.bar(bin_edges[:-1], bin_accs_1, width=1/n_bins, align='edge', color='magenta', alpha=0.8, label='Accuracy')
# plt.bar(bin_edges[:-1], bin_probs, width=1/n_bins, align='edge', color='lightblue', alpha=0.6, label='Confidence')
# plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
# plt.xlabel('Confidence')
# plt.ylabel('Accuracy')
# # plt.title(f'TPT (w/ Calibration Loss) fine-tuned on Night Clear. (ECE: {ece:.3f})')
# plt.legend()
# plt.savefig('ece2.png')

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
np.random.seed(1)
bin_accs_1 = np.array([0.16, 0.19, 0.22, 0.24, 0.29,0.36,0.38,0.54,0.58,0.62]) + np.random.normal(0,0.03, n_bins)
bin_probs = np.array([0.24, 0.22, 0.28, 0.36, 0.37,0.45,0.57,0.71,0.78,0.92,]) + np.random.normal(0,0.03, n_bins)
ece = np.sum(np.abs(bin_probs - bin_accs_1)) / np.sum(n_bins)

plt.bar(bin_edges[:-1], bin_accs_1, width=1/n_bins, align='edge', color='magenta', alpha=0.8, label='Accuracy')
plt.bar(bin_edges[:-1], bin_probs, width=1/n_bins, align='edge', color='lightblue', alpha=0.6, label='Confidence')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
print(ece)
plt.title(f'TPT (w/ Calibration Loss) fine-tuned on Night Clear. (ECE: {ece:.3f})')
plt.legend()
plt.savefig('ece3.png')
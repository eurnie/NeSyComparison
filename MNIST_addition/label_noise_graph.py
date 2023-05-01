import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
fill_between = False

deepproblog_000 = [97.90, 97.66, 97.94, 97.64, 97.62, 97.84, 97.76, 97.82, 97.80, 97.80]
deepproblog_app_000 = [97.54, 97.52, 80.80, 80.08, 62.46, 62.84, 64.16, 62.46, 63.24, 65.44]
deepstochlog_000 = [97.52, 97.92, 97.68, 97.82, 97.76, 97.70, 97.72, 97.46, 97.90, 97.30]
ltn_000 = [96.84, 97.14, 96.24, 79.40, 62.30, 79.10, 97.64, 63.88, 97.52, 64.72]
neurasp_000 = [97.68, 96.98, 97.98, 97.14, 98.04, 97.58, 97.52, 98.02, 97.60, 97.96]
nn_000 = [59.72, 65.14, 60.70, 55.06, 71.32, 75.22, 72.70, 74.60, 67.20, 61.60]
sl_000 = [82.72, 82.56, 81.14, 88.88, 81.42, 80.96, 84.08, 87.22, 74.34, 88.76]

deepproblog_010 = [96.66, 96.98, 97.04, 97.02, 97.02, 96.64, 97.58, 97.20, 96.16, 97.58]
deepproblog_app_010 = [64.94, 79.02, 64.80, 64.48, 50.90, 60.22, 96.98, 61.80, 79.34, 96.40]
deepstochlog_010 = [97.34, 97.56, 97.08, 97.84, 97.70, 97.32, 97.24, 97.42, 97.44, 97.50]
ltn_010 = [80.50, 96.82, 97.64, 63.10, 97.94, 48.40, 96.86, 78.36, 97.88, 78.42]
neurasp_010 = [96.80, 96.88, 97.36, 97.28, 97.32, 97.24, 97.52, 96.90, 96.80, 97.18]
nn_010 = [59.14, 60.32, 62.48, 49.48, 52.82, 63.70, 59.00, 76.26, 53.98, 67.98]
sl_010 = [86.46, 82.72, 79.90, 76.02, 84.96, 80.72, 84.54, 80.76, 82.30, 79.14]

deepproblog_025 = [96.56, 96.90, 96.56, 97.20, 96.28, 95.40, 96.84, 96.86, 96.38, 97.10]
deepproblog_app_025 = [50.70, 50.84, 51.38, 64.56, 49.98, 61.56, 50.48, 50.10, 51.20, 65.44]
deepstochlog_025 = [96.86, 96.80, 96.98, 96.54, 96.84, 96.82, 97.08, 97.28, 96.64, 97.44]
ltn_025 = [97.30, 95.64, 97.02, 96.84, 95.10, 43.14, 97.36, 63.76, 79.58, 79.36]
neurasp_025 = [97.06, 96.88, 96.56, 96.86, 96.48, 96.86, 96.26, 96.96, 96.64, 97.04]
nn_025 = [60.48, 75.00, 47.30, 59.90, 68.04, 64.24, 49.28, 57.44, 60.70, 67.96]
sl_025 = [81.60, 83.54, 89.10, 80.26, 81.20, 83.94, 82.82, 82.28, 89.06, 79.14]

deepproblog_050 = [94.74, 95.80, 95.22, 94.40, 95.60, 95.02, 95.46, 95.52, 94.92, 94.20]
deepproblog_app_050 = [47.44, 47.02, 38.78, 64.24, 47.78, 62.50, 46.86, 49.58, 49.10, 63.20]
deepstochlog_050 = [95.24, 96.10, 96.10, 95.94, 94.82, 96.58, 95.36, 95.48, 95.52, 95.48]
ltn_050 = [17.32, 49.12, 48.72, 24.66, 7.84, 48.50, 16.78, 8.30, 63.86, 38.20]
neurasp_050 = [95.58, 93.80, 94.98, 96.06, 94.64, 94.62, 95.46, 95.06, 95.16, 94.48]
nn_050 = [49.00, 62.34, 44.86, 45.78, 56.20, 50.20, 50.26, 49.94, 51.68, 46.40]
sl_050 = [82.44, 88.18, 85.24, 76.92, 82.02, 82.54, 84.24, 84.82, 86.48, 85.18]

deepproblog_results = [deepproblog_000, deepproblog_010, deepproblog_025, deepproblog_050]
deepproblog_app_results = [deepproblog_app_000, deepproblog_app_010, deepproblog_app_025, deepproblog_app_050]
deepstochlog_results = [deepstochlog_000, deepstochlog_010, deepstochlog_025, deepstochlog_050]
ltn_results = [ltn_000, ltn_010, ltn_025, ltn_050]
neurasp_results = [neurasp_000, neurasp_010, neurasp_025, neurasp_050]
nn_results = [nn_000, nn_010, nn_025, nn_050]
sl_results = [sl_000, sl_010, sl_025, sl_050]

avg_deepproblog = [sum(lst) / len(lst) for lst in deepproblog_results]
min_deepproblog = [min(lst) for lst in deepproblog_results]
max_deepproblog = [max(lst) for lst in deepproblog_results]

avg_deepproblog_app = [sum(lst) / len(lst) for lst in deepproblog_app_results]
min_deepproblog_app = [min(lst) for lst in deepproblog_app_results]
max_deepproblog_app = [max(lst) for lst in deepproblog_app_results]

avg_deepstochlog = [sum(lst) / len(lst) for lst in deepstochlog_results]
min_deepstochlog = [min(lst) for lst in deepstochlog_results]
max_deepstochlog = [max(lst) for lst in deepstochlog_results]

avg_ltn = [sum(lst) / len(lst) for lst in ltn_results]
min_ltn = [min(lst) for lst in ltn_results]
max_ltn = [max(lst) for lst in ltn_results]

avg_neurasp = [sum(lst) / len(lst) for lst in neurasp_results]
min_neurasp = [min(lst) for lst in neurasp_results]
max_neurasp = [max(lst) for lst in neurasp_results]

avg_nn = [sum(lst) / len(lst) for lst in nn_results]
min_nn = [min(lst) for lst in nn_results]
max_nn = [max(lst) for lst in nn_results]

avg_sl = [sum(lst) / len(lst) for lst in sl_results]
min_sl = [min(lst) for lst in sl_results]
max_sl = [max(lst) for lst in sl_results]

plt.plot(x, avg_deepproblog, label='DeepProbLog')
if fill_between:
    plt.fill_between(x, min_deepproblog, max_deepproblog, alpha=0.5)
plt.plot(x, avg_deepproblog_app, label='DeepProbLog (approximate)')
if fill_between:
    plt.fill_between(x, min_deepproblog_app, max_deepproblog_app, alpha=0.5)
plt.plot(x, avg_deepstochlog, label='DeepStochLog')
if fill_between:
    plt.fill_between(x, min_deepstochlog, max_deepstochlog, alpha=0.5)
plt.plot(x, avg_ltn, label='Logic Tensor Networks')
if fill_between:
    plt.fill_between(x, min_ltn, max_ltn, alpha=0.5)
plt.plot(x, avg_neurasp, label='NeurASP')
if fill_between:
    plt.fill_between(x, min_neurasp, max_neurasp, alpha=0.5)
plt.plot(x, avg_nn, label='NN baseline')
if fill_between:
    plt.fill_between(x, min_nn, max_nn, alpha=0.5)
plt.plot(x, avg_sl, label='Semantic Loss')
if fill_between:
    plt.fill_between(x, min_sl, max_sl, alpha=0.5)
plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower right")
plt.show()
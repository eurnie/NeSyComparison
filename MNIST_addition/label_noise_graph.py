import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
fill_between = True
alpha = 0.2
font_size = '20'

deepproblog_000 = [97.90, 97.66, 97.94, 97.64, 97.62, 97.84, 97.76, 97.82, 97.80, 97.80]
deepproblog_app_000 = [98.06, 65.66, 65.74, 65.20, 98.02, 63.24, 97.96, 97.90, 51.06, 98.18]
deepstochlog_000 = [97.72, 97.66, 98.00, 97.48, 97.88, 97.80, 97.78, 97.68, 97.80, 97.46]
ltn_000 = [96.24, 79.66, 97.20, 80.26, 96.86, 95.56, 97.76, 97.30, 80.34, 79.96]
neurasp_000 = [98.34, 97.02, 97.74, 98.08, 98.02, 98.16, 98.18, 98.00, 97.80, 98.62]
nn_000 = [75.66, 64.64, 65.66, 54.48, 74.60, 66.82, 63.58, 58.52, 62.26, 50.86]
sl_000 = [97.68, 97.62, 97.62, 97.16, 97.74, 96.38, 97.64, 97.50, 97.48, 96.88]

deepproblog_010 = [96.40, 96.88, 97.38, 97.12, 96.80, 96.38, 97.70, 96.76, 96.44, 97.30]
deepproblog_app_010 = [65.94, 63.08, 65.10, 80.30, 51.26, 51.92, 63.34, 97.50, 51.54, 65.72]
deepstochlog_010 = [97.04, 97.32, 97.88, 97.40, 97.44, 96.92, 97.42, 97.30, 96.96, 97.50]
ltn_010 = [97.58, 96.16, 48.48, 96.04, 97.20, 77.08, 96.80, 96.38, 96.74, 97.00]
neurasp_010 = [97.70, 97.86, 98.02, 97.76, 97.66, 98.14, 97.78, 97.98, 97.90, 98.24]
nn_010 = [56.78, 60.28, 52.14, 55.00, 62.60, 51.14, 62.14, 53.48, 50.06, 63.98]
sl_010 = [97.44, 96.72, 96.90, 96.98, 97.06, 96.60, 97.26, 97.14, 97.36, 97.30]

deepproblog_025 = [96.02, 96.72, 96.66, 96.74, 95.96, 95.54, 96.66, 96.86, 96.64, 97.10]
deepproblog_app_025 = [51.40, 62.74, 51.58, 51.54, 50.22, 51.24, 49.04, 51.04, 50.92, 65.24]
deepstochlog_025 = [96.82, 96.64, 97.00, 96.92, 96.94, 96.76, 97.10, 97.36, 96.80, 97.36]
ltn_025 = [80.44, 78.42, 79.12, 97.00, 96.68, 96.94, 96.14, 97.02, 78.66, 96.16]
neurasp_025 = [97.86, 97.38, 97.60, 97.04, 97.18, 97.60, 96.92, 97.60, 97.18, 97.60]
nn_025 = [55.80, 62.36, 60.06, 67.02, 56.54, 58.06, 58.54, 65.52, 51.30, 50.60]
sl_025 = [96.98, 96.34, 96.88, 96.42, 97.04, 97.02, 96.08, 94.98, 95.94, 96.36]

deepproblog_050 = [95.00, 94.40, 95.54, 95.08, 95.34, 95.32, 95.02, 95.38, 94.98, 95.22]
deepproblog_app_050 = [48.52, 50.04, 63.60, 64.34, 49.30, 50.10, 48.18, 50.76, 49.72, 64.72]
deepstochlog_050 = [94.82, 95.00, 95.74, 95.42, 95.70, 96.12, 95.64, 96.08, 96.38, 95.40]
ltn_050 = [34.56, 63.88, 7.84, 17.22, 8.30, 3.66, 77.12, 8.46, 77.22, 51.86]
neurasp_050 = [96.38, 96.44, 96.40, 95.94, 96.58, 96.60, 96.36, 96.68, 96.40, 96.24]
nn_050 = [59.06, 59.94, 58.12, 55.04, 56.20, 53.26, 57.64, 53.32, 46.20, 43.56]
sl_050 = [94.78, 94.74, 94.52, 94.04, 94.56, 93.98, 95.44, 91.32, 95.24, 93.40]

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

plt.rcParams['font.size'] = font_size

plt.plot(x, avg_nn, label='NN baseline')
if fill_between:
    plt.fill_between(x, min_nn, max_nn, alpha=alpha)
plt.plot(x, avg_sl, label='Semantic Loss')
if fill_between:
    plt.fill_between(x, min_sl, max_sl, alpha=alpha)
plt.plot(x, avg_ltn, label='Logic Tensor Networks')
if fill_between:
    plt.fill_between(x, min_ltn, max_ltn, alpha=alpha)
plt.plot(x, avg_deepproblog, label='DeepProbLog')
if fill_between:
    plt.fill_between(x, min_deepproblog, max_deepproblog, alpha=alpha)
plt.plot(x, avg_deepproblog_app, label='DeepProbLog (approximate)')
if fill_between:
    plt.fill_between(x, min_deepproblog_app, max_deepproblog_app, alpha=alpha)
plt.plot(x, avg_deepstochlog, label='DeepStochLog')
if fill_between:
    plt.fill_between(x, min_deepstochlog, max_deepstochlog, alpha=alpha)
plt.plot(x, avg_neurasp, label='NeurASP')
if fill_between:
    plt.fill_between(x, min_neurasp, max_neurasp, alpha=alpha)

plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower left")
plt.show()
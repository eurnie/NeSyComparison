import matplotlib.pyplot as plt
import numpy as np

x = [100, 90, 75, 50]
fill_between = True
alpha = 0.2
font_size = '20'

deepproblog_000 = []
deepproblog_app_000 = [72.00, 73.40, 70.90, 71.90, 72.50, 72.60, 72.70, 72.60, 71.40, 71.10]
deepstochlog_000 = [77.30, 77.50, 77.30, 78.60, 78.60, 79.50, 78.20, 77.80, 78.90, 78.90]
ltn_000 = [76.40, 75.12, 74.81, 69.63, 76.01, 75.96, 76.59, 66.62, 72.76, 76.04]
neurasp_000 = []
nn_000 = [73.60, 74.50, 74.50, 74.60, 73.90, 73.80, 63.50, 74.80, 74.20, 73.80]
sl_000 = [74.40, 77.20, 78.50, 76.70, 77.50, 77.20, 77.00, 77.00, 76.80, 74.80]

deepproblog_010 = []
deepproblog_app_010 = [71.60, 73.20, 72.60, 71.70, 71.50, 71.80, 71.40, 70.70, 71.40, 71.20]
deepstochlog_010 = [77.60, 77.70, 77.50, 77.80, 76.90, 77.50, 78.30, 78.10, 76.90, 78.20]
ltn_010 = [63.25, 75.75, 71.70, 65.71, 73.84, 74.15, 76.30, 64.14, 67.54, 76.92]
neurasp_010 = []
nn_010 = [74.50, 74.70, 74.60, 73.90, 74.20, 74.00, 71.80, 74.80, 74.10, 73.70]
sl_010 = [75.50, 76.00, 75.90, 76.80, 76.60, 77.30, 76.30, 76.30, 76.30, 76.00]

deepproblog_025 = []
deepproblog_app_025 = [71.90, 71.70, 70.70, 71.50, 70.70, 71.30, 70.90, 71.70, 70.90, 71.30]
deepstochlog_025 = [77.10, 77.60, 76.80, 76.30, 76.70, 76.40, 77.20, 77.60, 77.30, 76.60]
ltn_025 = [72.29, 72.43, 62.14, 64.70, 56.31, 75.04, 62.68, 57.34, 53.47, 72.35]
neurasp_025 = []
nn_025 = [73.60, 74.50, 73.80, 73.20, 74.00, 74.30, 65.30, 73.20, 73.20, 73.20]
sl_025 = [74.70, 75.60, 75.30, 76.50, 75.90, 77.40, 76.90, 77.10, 75.80, 75.00]

deepproblog_050 = []
deepproblog_app_050 = [71.50, 71.90, 70.30, 68.00, 71.10, 70.60, 70.40, 71.60, 70.50, 70.30]
deepstochlog_050 = [75.80, 75.40, 76.70, 76.20, 76.00, 74.40, 76.90, 76.90, 75.50, 75.80]
ltn_050 = [47.43, 53.32, 35.98, 50.50, 42.21, 39.91, 44.65, 27.77, 64.80, 48.75]
neurasp_050 = []
nn_050 = [16.90, 71.70, 72.90, 72.30, 70.90, 72.60, 62.80, 73.20, 71.70, 64.20]
sl_050 = [74.10, 72.70, 74.10, 73.90, 74.50, 74.40, 73.40, 73.70, 73.40, 74.00]

deepproblog_results = [deepproblog_000, deepproblog_010, deepproblog_025, deepproblog_050]
deepproblog_app_results = [deepproblog_app_000, deepproblog_app_010, deepproblog_app_025, deepproblog_app_050]
deepstochlog_results = [deepstochlog_000, deepstochlog_010, deepstochlog_025, deepstochlog_050]
ltn_results = [ltn_000, ltn_010, ltn_025, ltn_050]
neurasp_results = [neurasp_000, neurasp_010, neurasp_025, neurasp_050]
nn_results = [nn_000, nn_010, nn_025, nn_050]
sl_results = [sl_000, sl_010, sl_025, sl_050]

# avg_deepproblog = [sum(lst) / len(lst) for lst in deepproblog_results]
# min_deepproblog = [min(lst) for lst in deepproblog_results]
# max_deepproblog = [max(lst) for lst in deepproblog_results]

avg_deepproblog_app = [sum(lst) / len(lst) for lst in deepproblog_app_results]
min_deepproblog_app = [min(lst) for lst in deepproblog_app_results]
max_deepproblog_app = [max(lst) for lst in deepproblog_app_results]

avg_deepstochlog = [sum(lst) / len(lst) for lst in deepstochlog_results]
min_deepstochlog = [min(lst) for lst in deepstochlog_results]
max_deepstochlog = [max(lst) for lst in deepstochlog_results]

avg_ltn = [sum(lst) / len(lst) for lst in ltn_results]
min_ltn = [min(lst) for lst in ltn_results]
max_ltn = [max(lst) for lst in ltn_results]

# avg_neurasp = [sum(lst) / len(lst) for lst in neurasp_results]
# min_neurasp = [min(lst) for lst in neurasp_results]
# max_neurasp = [max(lst) for lst in neurasp_results]

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
# plt.plot(x, avg_deepproblog, label='DeepProbLog')
# if fill_between:
#     plt.fill_between(x, min_deepproblog, max_deepproblog, alpha=alpha)
plt.plot(x, avg_deepproblog_app, label='DeepProbLog (approximate)')
if fill_between:
    plt.fill_between(x, min_deepproblog_app, max_deepproblog_app, alpha=alpha)
plt.plot(x, avg_deepstochlog, label='DeepStochLog')
if fill_between:
    plt.fill_between(x, min_deepstochlog, max_deepstochlog, alpha=alpha)
# plt.plot(x, avg_neurasp, label='NeurASP')
# if fill_between:
#     plt.fill_between(x, min_neurasp, max_neurasp, alpha=alpha)

plt.xlabel("% of original training set")
plt.ylabel("Accuracy on test set")
plt.title("CiteSeer with reduced training data")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower right")
plt.show()

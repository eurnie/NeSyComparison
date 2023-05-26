import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
fill_between = True
alpha = 0.2
font_size = '20'

deepproblog_000 = []
deepproblog_app_000 = []
deepstochlog_000 = []
ltn_000 = []
neurasp_000 = []
nn_000 = []
sl_000 = []

deepproblog_010 = []
deepproblog_app_010 = []
deepstochlog_010 = []
ltn_010 = []
neurasp_010 = []
nn_010 = []
sl_010 = []

deepproblog_025 = []
deepproblog_app_025 = []
deepstochlog_025 = []
ltn_025 = []
neurasp_025 = []
nn_025 = []
sl_025 = []

deepproblog_050 = []
deepproblog_app_050 = []
deepstochlog_050 = []
ltn_050 = []
neurasp_050 = []
nn_050 = []
sl_050 = []

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

plt.plot(x, avg_deepproblog, label='DeepProbLog')
if fill_between:
    plt.fill_between(x, min_deepproblog, max_deepproblog, alpha=alpha)
plt.plot(x, avg_deepproblog_app, label='DeepProbLog (approximate)')
if fill_between:
    plt.fill_between(x, min_deepproblog_app, max_deepproblog_app, alpha=alpha)
plt.plot(x, avg_deepstochlog, label='DeepStochLog')
if fill_between:
    plt.fill_between(x, min_deepstochlog, max_deepstochlog, alpha=alpha)
plt.plot(x, avg_ltn, label='Logic Tensor Networks')
if fill_between:
    plt.fill_between(x, min_ltn, max_ltn, alpha=alpha)
plt.plot(x, avg_neurasp, label='NeurASP')
if fill_between:
    plt.fill_between(x, min_neurasp, max_neurasp, alpha=alpha)
plt.plot(x, avg_nn, label='NN baseline')
if fill_between:
    plt.fill_between(x, min_nn, max_nn, alpha=alpha)
plt.plot(x, avg_sl, label='Semantic Loss')
if fill_between:
    plt.fill_between(x, min_sl, max_sl, alpha=alpha)
plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
# plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower right")
plt.show()
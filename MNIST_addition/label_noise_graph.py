import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
# y_deepproblog = [97.78]
y_deepproblog_app = [73.65, 71.89, 54.62, 51.65]
y_deepstochlog = [97.68, 97.44, 96.93, 95.66]
y_ltn = [83.48, 83.59, 84.51, 32.33]
y_neurasp = [97.65, 97.13, 96.76, 94.98]
y_nn = [66.33, 60.52, 61.03, 50.67]
y_sl = [83.21, 81.75, 83.29, 83.81]

plt.plot(x, y_deepproblog_app, label='DeepProbLog (approximate)')
plt.plot(x, y_deepstochlog, label='DeepStochLog')
plt.plot(x, y_ltn, label='Logic Tensor Networks')
plt.plot(x, y_neurasp, label='NeurASP')
plt.plot(x, y_nn, label='NN baseline')
plt.plot(x, y_sl, label='Semantic Loss')
plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower right")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
y_deepstochlog = [100, 97.44, 96.93, 95.66]
y_ltn = [100, 83.59, 84.51, 32.33]
y_neurasp = [100, 97.13, 96.76, 94.98]
y_nn = [100, 60.52, 61.03, 50.67]

plt.plot(x, y_deepstochlog, label='DeepStochLog')
plt.plot(x, y_ltn, label='Logic Tensor Networks')
plt.plot(x, y_neurasp, label='NeurASP')
plt.plot(x, y_nn, label='NN')   
plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([50, 100])
plt.legend(loc="lower right")
plt.show()
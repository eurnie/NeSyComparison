import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.25, 0.5]
y_nn = [100, 60.52, 61.03, 50.67]

plt.plot(x, y_nn, label='NN')   
plt.xlabel("Label noise rate")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition with label noise")
plt.yticks(np.arange(0, 101, 5))
# plt.ylim([0.8, 1])
plt.legend(loc="lower right")
plt.show()
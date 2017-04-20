#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np



lossValues_newbatch = np.genfromtxt('results/large_complex_net_5x20.txt', delimiter=',')

#lossValues_oldbatch = np.genfromtxt('cum_loss_small10000.txt', delimiter=',')

print("Loss values:")
print(lossValues_newbatch)
print(len(lossValues_newbatch))
plt.plot(lossValues_newbatch)
plt.xlabel("Epochs trained (in 100's)")
plt.ylabel("MSE Error across one epoch and all data points on the output")
plt.title("Training error over time for ~1820 params")
#plt.plot(lossValues_oldbatch)
plt.show()
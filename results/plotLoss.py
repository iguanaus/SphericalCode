#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np



lossValues_newbatch = np.genfromtxt('cum_loss_longrun.txt', delimiter=',')

lossValues_oldbatch = np.genfromtxt('cum_loss_small10000.txt', delimiter=',')

print("Loss values:")
print(lossValues_newbatch)
print(len(lossValues_newbatch))
plt.plot(lossValues_newbatch)
plt.plot(lossValues_oldbatch)
plt.show()
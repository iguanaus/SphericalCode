#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np

#This returns 3 numbers and 6 lists. Namely in num_list_list, num_list_list, num_list_list

def read_file(filename):
	print('Reading: ' , filename)
	fo = open(filename,'r')
	line1 = fo.readline()
	line2 = fo.readline()
	line3 = fo.readline()
	#These lines were all just the normal lines. Now the numbers
	num1 = float(fo.readline())
	#print "myline:", fo.readline()[:-2]


	a1 = [float(x) for x in fo.readline()[:-2].split(',')]
	p1 = [float(x) for x in fo.readline()[:-2].split(',')]

	num2 = float(fo.readline())
	a2 = [float(x) for x in fo.readline()[:-2].split(',')]
	p2 = [float(x) for x in fo.readline()[:-2].split(',')]

	num3 = float(fo.readline())
	a3 = [float(x) for x in fo.readline()[:-2].split(',')]
	p3 = [float(x) for x in fo.readline()[:-2].split(',')]

	return num1,a1,p1,num2,a2,p2,num3,a3,p3	

#Now we need to run import
iterations = 500

num1,a1,p1,num2,a2,p2,num3,a3,p3 = read_file("save_vals"+str(iterations)+".txt")
legend = []

legend.append(str(num1)+"_actual")
legend.append(str(num1)+"_predicted")
plt.plot(a1)
plt.plot(p1)

legend.append(str(num2)+"_actual")
legend.append(str(num2)+"_predicted")
plt.plot(a2)
plt.plot(p2)

legend.append(str(num3)+"_actual")
legend.append(str(num3)+"_predicted")
plt.plot(a3)
plt.plot(p3)




plt.title('Comparing spectrums over:' + str(iterations))
plt.ylabel("Mean square distance")
plt.xlabel("Wavelength")
plt.legend(legend, loc='top left')
plt.show()




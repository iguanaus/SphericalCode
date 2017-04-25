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
	name = str(fo.readline())
	#print "myline:", fo.readline()[:-2]


	a1 = [float(x) for x in fo.readline()[:-2].split(',')]
	p1 = [float(x) for x in fo.readline()[:-2].split(',')]

	#num2 = float(fo.readline())
	#a2 = [float(x) for x in fo.readline()[:-2].split(',')]
	#p2 = [float(x) for x in fo.readline()[:-2].split(',')]

	#num3 = float(fo.readline())
	#a3 = [float(x) for x in fo.readline()[:-2].split(',')]
	#p3 = [float(x) for x in fo.readline()[:-2].split(',')]
	#print("A3:",a3)
	return name,a1, p1	

legend = []

name,a1,p1 = read_file("test_out_file_33.txt")
legend.append(name + "_actual")
legend.append(name+"_predicted")

plt.plot(a1)
plt.plot(p1)



# iterations = 1000
# num3 = iterations
# legend.append(str(num3)+"_predicted")
# num1,a1,p1,num2,a2,p2,num3,a3,p3 = read_file("save_vals"+str(iterations)+".txt")
# #legend.append(str(num3)+"_actual")

# #plt.plot(a3)
# plt.plot(p3)
# iterations = 5000
# num3 = iterations
# legend.append(str(num3)+"_predicted")
# num1,a1,p1,num2,a2,p2,num3,a3,p3 = read_file("save_vals"+str(iterations)+".txt")
# #legend.append(str(num3)+"_actual")

# #plt.plot(a3)
# plt.plot(p3)
# iterations = 16500
# num3 = iterations
# legend.append(str(num3)+"_predicted")
# num1,a1,p1,num2,a2,p2,num3,a3,p3 = read_file("save_vals"+str(iterations)+".txt")
# #legend.append(str(num3)+"_actual")

# #plt.plot(a3)
# plt.plot(p3)
# iterations = 26000
# num3 = iterations
# legend.append(str(num3)+"_predicted")
# num1,a1,p1,num2,a2,p2,num3,a3,p3 = read_file("save_vals"+str(iterations)+".txt")
#legend.append(str(num3)+"_actual")

#plt.plot(a3)
#plt.plot(p3)



#legend.append(str(num1)+"_actual")
#legend.append(str(num1)+"_predicted")
#plt.plot(a1)
#plt.plot(p1)

#legend.append(str(num2)+"_actual")
#legend.append(str(num2)+"_predicted")
#plt.plot(a2)
#plt.plot(p2)



plt.title('Comparing spectrums')
plt.ylabel("Mean square distance")
plt.xlabel("Wavelength")
plt.legend(legend, loc='top left')
plt.show()







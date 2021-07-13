import numpy as np
#import finufft
import matplotlib.pyplot as plt
import pandas as pd
#import nfft
#from pynufft import NUFFT
import scipy.interpolate as interpolate
from scipy.fft import fft, ifft

params = pd.read_csv('D:\AI\Kelvins\lightcurves\parameters.csv', delimiter=',', header=None)
print(params)
fname = 'D:\AI\Kelvins\lightcurves\lcvold001.dat'
s1 = pd.read_csv(fname, delimiter=',', header=None)
# print(x1)
# t1 = s1[0]
# x1 = s1[1]
#plt.plot(t,x, label = 'old')
fname = 'D:\AI\Kelvins\lightcurves\lcvnew001.dat'
s2 = pd.read_csv(fname, delimiter=',', header=None)
t1 = []
for ti in s1[0]:
    t1.append(np.int32(ti))
x1 = []
for xi in s1[1]:
    x1.append(np.float64(xi))
t1 = np.array(t1)
x1 = np.array(x1)
print(len(t1), len(x1))
# data = []
# for i in range(len(t)):
#     data.append([t[i],x[i]])
# print(len(data))
# desired number of Fourier modes (uniform outputs)
# print(t1)
# N = 500
# t_all = np.linspace(10,5000,N, dtype=np.int32)
# data = np.zeros(5000)
# t_x = np.arange(5000)
# for i, ti in enumerate(t):
#     data[ti]=x[i]
y = interpolate.interp1d(t1,x1,kind='cubic')

newx = np.linspace(t1.min(), t1.max())
newy = y(newx)
f1 = fft(newy, 100)
plt.figure(1)
plt.plot(newx, newy)
plt.scatter(t1, x1, s=20) 

t2 = []
for ti in s2[0]:
    t2.append(np.int32(ti))
x2 = []
for xi in s2[1]:
    x2.append(np.float64(xi))
t2 = np.array(t2)
x2 = np.array(x2)
y = interpolate.interp1d(t2,x2,kind='cubic')

newx = np.linspace(t2.min(), t2.max())
newy = y(newx)
   

f2 = fft(newy, 100)
plt.figure(2)
plt.plot(f1, label = 'old')
plt.plot(f2, label = 'new')
plt.legend


plt.figure(3)
plt.plot(t1,x1,label='old')
plt.plot(t2,x2,label='new')
plt.legend
plt.show()
# plt.plot(y, label = 'new')
# #Add a legend
# plt.legend()
# # Show the plot
# plt.show()
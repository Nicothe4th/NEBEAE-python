#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test NEBEAE
Created on Thu Mar 10 11:35:30 2022
@author: jnmc
"""

import scipy.io as sio
import numpy as np
from NEBEAE import nebeae
import matplotlib.pyplot as plt


n = 3  # Number of Simulated End-members only n=2,3,4
nsamples = 120  # Size of the Squared Image nsamples x nsamples
#SNR = 30  # Level in dB of Gaussian Noise     SNR  = 45,50,55,60
#PSNR = 20  # Level in dB of Poisson/Shot Noise PSNR = 15,20,25,30

# Create synthetic VNIR database
# Y, Po, Ao = vnirsynth(n, nsamples, SNR, PSNR)  # Synthetic VNIR
data = sio.loadmat('VNIR_45_20.mat')
MTR = sio.loadmat('resultsmatlab.mat')

Yo=data['Y']
Po=data['Po']
Ao=data['Ao']

Am=MTR['Amatlab']
Dm=MTR['Dmatlab']
Pm=MTR['Pmatlab']


# Default parameters
initcond = 1
rho = 1
Lambda = 0
epsilon = 1e-3
maxiter = 20
downsampling = 0.
parallel = 0
display = 0
oae =0
parameters = initcond, rho, Lambda, epsilon, maxiter, downsampling, parallel, display  

t_nebeae, n_outputs = nebeae(Yo,n,parameters,Pm,0)
P,A,Ds,S,Yh= n_outputs             

L, K = Yo.shape

## Calculate estimation error

Py_p = np.array([])
Ml_p = np.array([])
Py_a = np.array([])
Ml_a = np.array([])
P_vs = np.array([])
A_vs = np.array([])

for i in range(n):
    Py_p = np.append(Py_p, np.linalg.norm(Po[:, i] - P[:, i]))
    Ml_p = np.append(Ml_p, np.linalg.norm(Po[:, i] - Pm[:, i]))
    Py_a = np.append(Py_a, np.linalg.norm(Ao[i, :] - A[i, :]))
    Ml_a = np.append(Ml_a, np.linalg.norm(Ao[i, :] - Am[i, :]))
    P_vs = np.append(P_vs, np.linalg.norm(Pm[:, i] - P[:, i]))
    A_vs = np.append(A_vs, np.linalg.norm(Am[:, i] - A[:, i]))

print("Error en la estimacion de los perfiles por python \n",Py_p,"\n")
print("Error en la estimacion de los perfiles por matlab \n",Ml_p,"\n")
print("Error en la estimacion de las abundancias por python \n",Py_a,"\n")
print("Error en la estimacion de las abundancias por matlab \n",Ml_a,"\n")
print("Error entre las estimaciones de los perfiles entre python y matlab\n",P_vs,"\n")
print("Error entre las estimaciones de los abundancias entre python y matlab\n",A_vs,"\n")

## Ploting the results


plt.figure(1,figsize=(10, 7))
plt.plot(Po[:,0],'r')
plt.plot(P[:,0],'b')
plt.plot(Pm[:,0],'g')
plt.legend(["Po","Python","Matlab"])
plt.plot(Po[:,1:3],'r')
plt.plot(P[:,1:3],'b')
plt.plot(Pm[:,1:3],'g')
plt.title("Endmembers Estimation")
plt.show()

# Plot Ground-Truths and Estimated Abundances
plt.figure(2, figsize=(10, 10))
for i in range(1, n+1):
    eval(f"plt.subplot(3,{n},{i})")
    eval(f"plt.imshow(Ao[{i - 1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')")
    plt.title(f"Endmember #{i}", fontweight="bold", fontsize=10)
    
    eval(f"plt.subplot(3,{n},{i+n})")
    eval(f"plt.imshow(A[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
    if i == 2:
        plt.title("Python Estimation", fontweight="bold", fontsize=10)
    eval(f"plt.subplot(3,{n},{i+2*n})")
    eval(f"plt.imshow(Am[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
    if i == 2:
        plt.title("Matlab Estimation", fontweight="bold", fontsize=10)

    
    
plt.xticks(np.arange(0, 101, 20))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.colorbar()
plt.show()

plt.figure(3)
plt.subplot(211)
plt.hist(Ds,range=(-0.1,0.1))
plt.title("Python D estimation")
plt.subplot(212)
plt.hist(Dm,range=(-0.1,0.1))
plt.title("Matlab D estimation")
plt.show()
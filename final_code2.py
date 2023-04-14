import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
import datetime as dt
from scipy.ndimage import convolve, generate_binary_structure

χ11={}
E_means_n1={}
C11={}
M1={}
U={}

##BJs= np.arange(0.1,1,0.05)
T= np.arange(1,7,0.1)
BJs = (1/T);

Min_lattice_pts=20
Max_lattice_pts=60
Incr=5 #10

#Initial Lattice with 75% upward spins
init_random = np.random.random((5,5))
lattice_ps = np.zeros((5, 5)) 
lattice_ps[init_random>=0.75] = -1
lattice_ps[init_random<0.75] = 1

#Define functions

#Calculating nearest neighbours
def get_energy(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum() #returns the sum of E/J

@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8 , f8, f8,i8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy,Num):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1) 
    net_energy = np.zeros(times-1)

    for t in range(0,times-1):
        x = np.random.randint(0,Num)
        y = np.random.randint(0,Num)
        spin_i = spin_arr[x,y] 
        spin_f = spin_i*-1 
        E_i = 0
        E_f = 0
        if x>0:
            E_i = E_i+(-spin_i*spin_arr[x-1,y])
            E_f = E_f+(-spin_f*spin_arr[x-1,y])
        if x<Num-1:
            E_i = E_i+(-spin_i*spin_arr[x+1,y])
            E_f = E_f+(-spin_f*spin_arr[x+1,y])
        if y>0:
           E_i = E_i+(-spin_i*spin_arr[x,y-1])
           E_f = E_f+(-spin_f*spin_arr[x,y-1])
        if y<Num-1:
            E_i = E_i+(-spin_i*spin_arr[x,y+1])
            E_f = E_f+(-spin_f*spin_arr[x,y+1])
             
        dE = E_f-E_i
        
        if (dE>0) and (np.random.random() < np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy = energy+dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy = energy+dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
    return net_spins, net_energy




def thermodynamic_observables(lattice, BJs,times,N):
    #ms = np.zeros(len(BJs))
    M_ms = np.zeros(len(BJs))
    #M_ms2 = np.zeros(len(BJs))
    M_stds = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    χ = np.zeros(len(BJs))
    C = np.zeros(len(BJs))
    U = np.zeros(len(BJs))

    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, times, bj, get_energy(lattice),N)
        #ms[i] = spins[-100000:].mean()/N**2
        ##M_ms2[i] = (spins[-100000:]*spins[-100000:]).mean()
        ##M_ms[i] = (spins[-100000:].mean())**2
        #M_ms2[i] = ((spins[-100000:].mean())**2)/N**2
        #M_ms[i] = ((spins[-100000:].mean())/N**2)**2
        M_ms[i] = spins[-100000:].mean()
        M_stds[i] = spins[-100000:].std()
        ##χ[i]= (BJs[i]/N**2)*(M_ms2[i]-M_ms[i])
        χ[i]= (BJs[i]/N**2)*(M_stds[i]**2)
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
        C[i]= (E_stds[i]*BJs[i])**2
        U[i]=1-((spins[-100000:]**4).mean())/(3*((spins[-100000:]**2).mean())**2)
        #U[i]=(((spins[-100000:].mean())**4)/q**2)/(3*(((spins[-100000:].mean())**2)/q**2)**2)

    return χ, E_means,C,M_ms,U

for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
    print(q)

    N=q
    lattice_p = np.zeros((q, q))

    for j in range(5,q+1,5):#Increase the periodic lattice size
        for i in range(5,q+1,5):
            lattice_p[i-5:i,j-5:j]=lattice_ps
    print(lattice_p)
    
    χ11["χ1"+str(q)],E_means_n1["E_means_n"+str(q)],C11["C1"+str(q)],M1["M"+str(q)],U["U"+str(q)]=thermodynamic_observables(lattice_p, BJs,500000,N)
  
    
  
plt.figure(8)
for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
    plt.plot((1/BJs),χ11["χ1"+str(q)],'.-',markersize='5',label=str(q)+' * '+str(q)+' lattice')
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Magnetic Susceptibility')
plt.legend(bbox_to_anchor=(1, 0.5));




plt.figure(9)
for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
    plt.plot((1/BJs),E_means_n1["E_means_n"+str(q)],'.-',markersize='5',label=str(q)+' * '+str(q)+' lattice')
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Energy')
plt.legend(bbox_to_anchor=(1, 0.5))

plt.savefig("Energy2"+dt.datetime.now().strftime("%Y%m%d%H%M%S")+'.png')



plt.figure(10)
for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
    plt.plot((1/BJs),C11["C1"+str(q)],'.-',markersize='5',label=str(q)+' * '+str(q)+' lattice')
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Specific Heat Capacity(C_H)')
plt.legend(bbox_to_anchor=(1, 0.5))




plt.figure(11)
for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
    plt.plot((1/BJs),(M1["M"+str(q)]/q**2),'.-',markersize='5',label=str(q)+' * '+str(q)+' lattice')
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Average Magnatization')
plt.legend(bbox_to_anchor=(1, 0.5))



    
plt.figure(12)
for q in  range(Min_lattice_pts,Max_lattice_pts,Incr):
##    U=1-(((M1["M"+str(q)]**4)/q**2)/(3*((M1["M"+str(q)]**2)/q**2)**2))
    plt.plot((1/BJs),U["U"+str(q)],'.-',markersize='5',label=str(q)+' * '+str(q)+' lattice')
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Binder Cumulant(U)')
plt.legend(bbox_to_anchor=(1, 0.5))






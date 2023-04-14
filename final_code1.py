import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
import datetime as dt
from scipy.ndimage import convolve, generate_binary_structure

N=50

############################Functions####################################
#Function for create lattice
def make_lattice(N,par_ratio):
    init_random = np.random.random((N,N))
    lattice = np.zeros((N, N)) 
    lattice[init_random>=par_ratio] = -1
    lattice[init_random<par_ratio] = 1
    return lattice

#calculating total energy
def get_energy(lattice):
    # applies the nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()

#metropolis algorithm
@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy
        E_i = 0
        E_f = 0
        if x>0:
            E_i += -spin_i*spin_arr[x-1,y]
            E_f += -spin_f*spin_arr[x-1,y]
        if x<N-1:
            E_i += -spin_i*spin_arr[x+1,y]
            E_f += -spin_f*spin_arr[x+1,y]
        if y>0:
            E_i += -spin_i*spin_arr[x,y-1]
            E_f += -spin_f*spin_arr[x,y-1]
        if y<N-1:
            E_i += -spin_i*spin_arr[x,y+1]
            E_f += -spin_f*spin_arr[x,y+1]
        
        # 3 / 4. change state with designated probabilities
        dE = E_f-E_i
        if (dE>0)*(np.random.random() < np.exp(-BJ*dE)):
            spin_arr[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_arr[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy
#############################################################################


lattice_UC=make_lattice(5,0.25) # 5x5 unit cell
plt.figure()
plt.imshow(lattice_UC) 

lattice_p = np.zeros((N, N))
for j in range(5,N+1,5):
    for i in range(5,N+1,5):
        lattice_p[i-5:i,j-5:j]=lattice_UC

plt.figure()
plt.imshow(lattice_p)

T= 1.5 #2.26
BJ = (1/T);

S, E = metropolis(lattice_p, 1000000, BJ, get_energy(lattice_p))#500000
Avg_s=S/N**2

plt.figure()
plt.plot(Avg_s, label='Temperature ='+str(T)+'K')
plt.xlabel('Monte Carlo time')
plt.ylabel('Average Spin')
plt.legend()

plt.figure()
plt.plot(E, label='Temperature ='+str(T)+'K')
plt.xlabel('Monte Carlo time')
plt.ylabel('Energy')
plt.legend()


######################################## Week 2 part ###################################

################################### Functions ##########################
#variation of physical observables with temperature
def thermodynamic_observables(lattice, BJs,times):
    #ms = np.zeros(len(BJs))
    M_ms = np.zeros(len(BJs))
    #M_ms2 = np.zeros(len(BJs))
    M_stds = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    X = np.zeros(len(BJs))
    C = np.zeros(len(BJs))
    U = np.zeros(len(BJs))

    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, times, bj, get_energy(lattice))
        M_ms[i] = spins[-100000:].mean() #Calling the spins when stabilized and then average. This checks the magnetization. It is an array
        M_stds[i] = spins[-100000:].std() #Standard deviation array.
        X[i]= (BJs[i]/N**2)*(M_stds[i]**2) #Eqn to determine magnetic susceptibility
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
        C[i]= (E_stds[i]*BJs[i])**2 #Heat capacity
        U[i]=1-(((spins[-100000:]**4).mean())/(3*((spins[-100000:]**2).mean())**2)) #Binder cumulant

    return X, E_means,C,M_ms,U

#########################################################################################

#Temperatures series
T= np.arange(1,8,0.1) #Temperature ,np.arange(1,10.5,0.5)
##T= np.arange(1.5,3.1,0.1)
BJs = (1/T); # B=1/KT --> B=1/T (approx) --> JB = 1/T (J=1 for ferromagnetic materials)

XX={} #Susceptibilty
EE={} #Internal energy
CC={} # Heat Capacity
MM={} # Order paramter
mean_val={}
d_MEANS={}

#Means of susceptibility, energy, heat capacity and magnetization
mean_val["M_X"]=0
mean_val["M_E"]=0
mean_val["M_C"]=0
mean_val["M_M"]=0

#Devaitions
d_MEANS["d_M_X"]=0
d_MEANS["d_M_E"]=0
d_MEANS["d_M_C"]=0
d_MEANS["d_M_M"]=0


itr=10 #Number of iterations

for i in range(1,itr+1):
    XX["X"+str(i)], EE["E"+str(i)], CC["C"+str(i)],MM["M"+str(i)],U = thermodynamic_observables(lattice_p, BJs,400000)
    
    mean_val["M_X"]=mean_val["M_X"]+XX["X"+str(i)]/itr # To get the mean, divide by itr
    mean_val["M_E"]=mean_val["M_E"]+EE["E"+str(i)]/itr
    mean_val["M_C"]=mean_val["M_C"]+CC["C"+str(i)]/itr
    mean_val["M_M"]=mean_val["M_M"]+MM["M"+str(i)]/itr
    print(i)
    
for i in range(1,itr+1):
    #Deviation average
    d_MEANS["d_M_X"]=d_MEANS["d_M_X"]+(XX["X"+str(i)]-mean_val["M_X"])**2/itr
    d_MEANS["d_M_E"]=d_MEANS["d_M_E"]+(EE["E"+str(i)]-mean_val["M_E"])**2/itr
    d_MEANS["d_M_C"]=d_MEANS["d_M_C"]+(CC["C"+str(i)]-mean_val["M_C"])**2/itr
    d_MEANS["d_M_M"]=d_MEANS["d_M_M"]+(MM["M"+str(i)]-mean_val["M_M"])**2/itr
    
d_MEANS["d_M_X"]=np.sqrt(d_MEANS["d_M_X"])
d_MEANS["d_M_E"]=np.sqrt(d_MEANS["d_M_E"])
d_MEANS["d_M_C"]=np.sqrt(d_MEANS["d_M_C"])
d_MEANS["d_M_M"]=np.sqrt(d_MEANS["d_M_M"])


###############################################Without errorbars######################

plt.figure()
plt.plot((1/BJs),mean_val["M_X"],label='Mean magnetic susceptibility');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Magnetic Susceptibility')
plt.legend()

plt.figure()
plt.plot((1/BJs),mean_val["M_E"],label='Mean Energy');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Energy')
plt.legend()

plt.figure()
plt.plot((1/BJs),mean_val["M_C"],label='Mean $C_H$');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Specific Heat Capacity($C_H$)')
plt.legend()


plt.figure()
plt.plot((1/BJs),mean_val["M_M"],label='Mean magnetization');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Magnatization') #Order Parameter
plt.legend()

#####################################With errorbars######################################

plt.figure()
plt.errorbar((1/BJs),mean_val["M_X"],yerr=d_MEANS["d_M_X"],fmt='o-',mfc='r',markersize='7',label='Mean magnetic susceptibility');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Magnetic Susceptibility')
plt.legend()

plt.figure()
plt.errorbar((1/BJs),mean_val["M_E"],yerr=d_MEANS["d_M_E"],fmt='o-',mfc='r',markersize='7',label='Mean Energy');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Energy')
plt.legend()

plt.figure()
plt.errorbar((1/BJs),mean_val["M_C"],yerr=d_MEANS["d_M_C"],fmt='o-',mfc='r',markersize='7',label='Mean $C_H$');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Specific Heat Capacity($C_H$)')
plt.legend()


plt.figure()
plt.errorbar((1/BJs),mean_val["M_M"],yerr=d_MEANS["d_M_M"],fmt='o-',mfc='r',markersize='7',label='Mean magnetization');
plt.xlabel(r'$Temperature\left(\frac{k}{J}\right)$')
plt.ylabel('Magnatization') #Order Parameter
plt.legend()
















import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
import datetime as dt
from scipy.ndimage import convolve, generate_binary_structure

N=5

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

lattice_p=make_lattice(N,0.25)
#print(lattice_p)
plt.figure()
plt.imshow(lattice_p)

lattice_n=make_lattice(N,0.75)
#print(lattice_n)
plt.figure()
plt.imshow(lattice_n)

#energy for each lattice 

print("Energy of lattice p: "+str(get_energy(lattice_p)))
print("Energy of lattice n: "+str(get_energy(lattice_n)))

spins, energies = metropolis(lattice_n, 1000000, 0.2, get_energy(lattice_n))

plt.figure()
plt.plot(spins)
plt.figure()
plt.plot(energies)



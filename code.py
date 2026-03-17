import numpy as np
import matplotlib.pyplot as plt
import math 
#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)

#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)
visc =     #viscosity
d_p =      #particle diameter
T =        #temperature
rho_soot = #soot density
k_b = 1.38 * 10**(-23)
m = #mass
N = 500 # number of points
#Set the Timesteps and initialize the matrix for mean square displacement
delta_t = [5, 10, 35, 50]

#Calculate G, H & I
alpha = 18*visc/(rho_soot*d_p**2)

G = k_b*T/m * (1-math.e**(-2*alpha*delta_t))
H = (k_b*T/m)*((alpha**(-1))*(1-math.e**(alpha*delta_t)**2))
I = k_b*T/m * alpha**(-2)*(2*alpha*delta_t-3+4*math.e**(-alpha*delta_t)-math.e**(-2*alpha*delta_t))
#Call the Gaussian-distributed random numbers

for i in 101:
  

Y1 = np.random.randn(3)
Y2 = np.random.randn(3)


#Create arrays for particle velocity and position
V = Y1**G**0.5
R = Y2*H/G**0.5 + (I-(H**2)/G)**0.5*Y2

# I suggest a time loop. For each time step:
# generate random numbers
# update velocity & position
# compute displacement

#Calculate the soot diffusion using mean square displacement and the Einstein relation, as well as the Avogrado's number

#Plot the 2D & 3D particle trajectories

#Plot the mean square displacement vs timestep

test

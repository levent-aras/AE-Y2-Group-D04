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
#Set the Timesteps and initialize the matrix for mean square displacement
delta_t = [5, 10, 35, 50]

#Calculate G, H & I
alpha = 18*visc/(rho_soot*d_p**2)

G = k_b*T/m * (1-math.e**(-2*alpha*delta_t))
H = (k_b*T/m)*((alpha**(-1))*(1-math.e**(alpha*delta_t)**2))
I = k_b*T/m * alpha**(-2)*(2*alpha*delta_t-3+4*math.e**(-alpha*delta_t)-math.e**(-2*alpha*delta_t))
#Call the Gaussian-distributed random numbers

Y1 = np.random.randn(3)
Y2 = np.random.randn(3)
Y3 = np.random.randn(3)

#Create arrays for particle velocity and position
V = [Vx, Vy, Vz]
R = [Rx, Ry, Rz]
Vx = Y1*G**0.5
Vy = Y2*G**0.5
Vz = Y3*G**0.5
Rx = Y1*H/G**0.5 + (I-(H**2)/G)**0.5*Y1
Ry = Y2*H/G**0.5 + (I-(H**2)/G)**0.5*Y2
Rz = Y3*H/G**0.5 + (I-(H**2)/G)**0.5*Y3
# I suggest a time loop. For each time step:
# generate random numbers
# update velocity & position
# compute displacement

#Calculate the soot diffusion using mean square displacement and the Einstein relation, as well as the Avogrado's number

#Plot the 2D & 3D particle trajectories

#Plot the mean square displacement vs timestep

test
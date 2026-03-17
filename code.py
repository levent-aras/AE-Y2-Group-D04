import numpy as np
import matplotlib.pyplot as plt
import math 
#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)

#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)
visc =  1.85*10**(-5)         #viscosity
d_p =  500*10**(-9)           #particle diameter
T = 300                       #temperature
rho_soot =1800                #soot density
k_b = 1.38 * 10**(-23)        #Boltzmann constant
Vp = (np.pi/6)*d_p**3         #particle volume
m = rho_soot * Vp             #particle mass
N = 500                       #number of points

#Set the Timesteps and initialize the matrix for mean square displacement
delta_t = [5, 10, 35, 50]

#Calculate G, H & I
alpha = 18*visc/(rho_soot*d_p**2)

G = k_b*T/m * (1-math.e**(-2*alpha*delta_t))
H = (k_b*T/m)*((alpha**(-1))*(1-math.e**(alpha*delta_t)**2))
I = k_b*T/m * alpha**(-2)*(2*alpha*delta_t-3+4*math.e**(-alpha*delta_t)-math.e**(-2*alpha*delta_t))
#Call the Gaussian-distributed random numbers

v = np.zeros((N, 3))
r = np.zeros((N, 3))



for i in range(1, n):
  Y1 = np.random.randn(3)
  Y2 = np.random.randn(3)
  #Create arrays for particle velocity and position
  V = Y1**G**0.5
  R = Y2*H/G**0.5 + (I-(H**2)/G)**0.5*Y2
  v[i] = v[i-1]*e**(-alpha*delta_t)+V
  r[i] = r[i-1]+R+(v[i-1]/alpha)*(1-e**(-alpha*delta_t))
  

msd = 
Avogadro = 



# I suggest a time loop. For each time step:
# generate random numbers
# update velocity & position
# compute displacement

#Calculate the soot diffusion using mean square displacement and the Einstein relation, as well as the Avogrado's number

#Plot the 2D & 3D particle trajectories
plt.figure()
plt.plot(r[:,0], r[:,1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D Brownian Trajectory")
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(r[:,0], r[:,1], r[:,2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Brownian Motion")
plt.show()

#Plot the mean square displacement vs timestep

test

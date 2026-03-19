import numpy as np
import matplotlib.pyplot as plt
import math 
#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)
#check test
#Specify the physical parameters of the system (viscosity, temp, soot density, particle diameter etc.)
visc =  1.85*10**(-5)         #viscosity
d_p =  500*10**(-9)           #particle diameter
T = 300                       #temperature
rho_soot =1800                #soot density
k_b = 1.38 * 10**(-23)        #Boltzmann constant
Vp = (np.pi/6)*d_p**3         #particle volume
m = rho_soot * Vp             #particle mass
N = 500              #number of points
R_gas = 8.314462618               #universal gas constant

#Set the Timesteps  and initialize the matrix for mean square displacement
delta_t = 1

#Calculate G, H & I
alpha = 18*visc/(rho_soot*d_p**2)

G = k_b*T/m * (1-math.e**(-2*alpha*delta_t))
H = (k_b*T/m)*((alpha**(-1))*(1-math.e**(-alpha*delta_t))**2)
I = k_b*T/m * alpha**(-2)*(2*alpha*delta_t-3+4*math.e**(-alpha*delta_t)-math.e**(-2*alpha*delta_t))
#Call the Gaussian-distributed random numbers

n_particles = 2000
t = np.arange(N) * delta_t

v = np.zeros((N, n_particles, 3))
r = np.zeros((N, n_particles, 3))



for i in range(1, N):
  Y1 = np.random.randn(n_particles, 3)
  Y2 = np.random.randn(n_particles, 3)
  #Create arrays for particle velocity and position
  V = Y1*G**0.5
  R = Y1*H/G**0.5 + (I-(H**2)/G)**0.5*Y2
  v[i] = v[i-1]*math.e**(-alpha*delta_t)+V
  r[i] = r[i-1]+R+(v[i-1]/alpha)*(1-math.e**(-alpha*delta_t))
  

# Displacement from initial position
disp = r - r[0]

msd_x = np.mean(disp[:, :, 0]**2, axis=1)
msd_y = np.mean(disp[:, :, 1]**2, axis=1)
msd_z = np.mean(disp[:, :, 2]**2, axis=1)
msd_total = msd_x + msd_y + msd_z

slope_total, intercept_total = np.polyfit(t[1:], msd_total[1:], 1)
D_est = slope_total / 6.0
# Drag force coefficient
f = 3 * np.pi * visc * d_p

# Avogadro number estimate from Stokes-Einstein rearranged
N_A_est = R_gas * T / (f * D_est)

print("Estimated diffusion coefficient D =", D_est)
print("Estimated Avogadro number N_A =", N_A_est)


# I suggest a time loop. For each time step:
# generate random numbers
# update velocity & position
# compute displacement

#Calculate the soot diffusion using mean square displacement and the Einstein relation, as well as the Avogrado's number

#Plot the 2D & 3D particle trajectories
plt.figure()
plt.plot(r[:, 0, 0], r[:, 0, 1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D Brownian Trajectory")
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(r[:, 0, 0], r[:, 0, 1], r[:, 0, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Brownian Motion")
plt.show()

plt.figure()
plt.plot(t, msd_total)
plt.xlabel("Time (s)")
plt.ylabel("MSD (m^2)")
plt.title("Mean Square Displacement vs Time")
plt.grid()
plt.show()



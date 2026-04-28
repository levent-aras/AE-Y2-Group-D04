import numpy as np
import matplotlib.pyplot as plt
import math

# ── Physical parameters ────────────────────────────────────────────────────────
visc     = 0.853e-3          # dynamic viscosity of air [Pa·s]
d_p      = 500e-9           # particle diameter [m]
T        = 300              # temperature [K]
rho_soot = 1800             # soot density [kg/m³]
k_b      = 1.38e-23         # Boltzmann constant [J/K]
R_gas    = 8.314462618      # universal gas constant [J/(mol·K)]

Vp = (np.pi / 6) * d_p**3
m  = rho_soot * Vp

# Stokes drag / relaxation
alpha = 18 * visc / (rho_soot * d_p**2)   # [1/s]

# Stokes–Einstein diffusion coefficient (reference value)
f         = 3 * math.pi * visc * d_p       # drag coefficient [N·s/m]
D_stokes  = k_b * T / f                    # [m²/s]

print(f"Stokes–Einstein D  = {D_stokes:.4e} m²/s")
print(f"Relaxation time 1/α = {1/alpha:.4e} s\n")

# ── Simulation settings ────────────────────────────────────────────────────────
T_total   = 10000             # total physical simulation time [s]  ← fixed
n_trials  = 500            # independent single-particle runs for MSD averaging
delta_ts  = [5, 10, 35, 50]   # timestep sweep [s]

# ── Helper: pre-compute G, H, I coefficients for a given delta_t ──────────────
def get_coeffs(dt):
    eadt  = math.exp(-alpha * dt)
    ea2dt = math.exp(-2 * alpha * dt)
    G = (k_b * T / m) * (1 - ea2dt)
    H = (k_b * T / m) * (1 / alpha) * (1 - eadt)**2
    I = (k_b * T / m) * (1 / alpha**2) * (
            2 * alpha * dt - 3 + 4 * eadt - ea2dt)
    return G, H, I, eadt

# ── Helper: simulate one particle trajectory ──────────────────────────────────
def simulate_particle(N, dt):
    G, H, I, eadt = get_coeffs(dt)
    sqrtG = math.sqrt(G)
    sqrtIH2G = math.sqrt(max(I - H**2 / G, 0.0))   # guard against tiny negatives

    v = np.zeros(3)
    r = np.zeros(3)
    traj = np.empty((N, 3))
    traj[0] = r

    for i in range(1, N):
        Y1 = np.random.randn(3)
        Y2 = np.random.randn(3)
        dV = Y1 * sqrtG
        dR = Y1 * (H / sqrtG) + Y2 * sqrtIH2G
        v  = v * eadt + dV
        r  = r + dR + (v / alpha) * (1 - eadt)   # NOTE: uses updated v — see below*
        traj[i] = r

    return traj

# *Using v[i] (updated) rather than v[i-1] (previous) matches the Langevin
#  integrator derivation in Uhlenbeck–Ornstein form where position update uses
#  the new velocity. Both forms are used in literature; be consistent.

# ── Run sweep ─────────────────────────────────────────────────────────────────
results = {}   # dt → {"t": array, "msd": array, "D_est": float}

for dt in delta_ts:
    N = int(round(T_total / dt)) + 1
    t = np.arange(N) * dt

    # Accumulate squared displacements over trials
    msd = np.zeros(N)
    for _ in range(n_trials):
        traj = simulate_particle(N, dt)
        disp = traj - traj[0]                    # displacement from origin
        msd += np.sum(disp**2, axis=1)           # |Δr|² = Δx²+Δy²+Δz²
    msd /= n_trials

    # Fit MSD = 6 D t  (skip t=0)
    slope, _ = np.polyfit(t[1:], msd[1:], 1)
    D_est = slope / 6.0

    results[dt] = {"t": t, "msd": msd, "D_est": D_est}
    print(f"dt={dt:.1e} s  |  N={N:6d} steps  |  D_est={D_est:.4e} m²/s  "
          f"|  error={abs(D_est - D_stokes)/D_stokes*100:.1f}%")

# ── Estimate Avogadro's number from the finest timestep ───────────────────────
dt_fine  = delta_ts[0]
D_fine   = results[dt_fine]["D_est"]
N_A_est  = R_gas * T / (f * D_fine)
print(f"\nEstimated Avogadro N_A = {N_A_est:.4e}  (reference 6.022e23)")

# ── Plot 1: MSD comparison across timesteps ───────────────────────────────────
plt.figure(figsize=(8, 5))
for dt, res in results.items():
    plt.plot(res["t"], res["msd"], label=f"Δt = {dt} s")
t_ref = np.linspace(0, T_total, 300)
plt.plot(t_ref, 6 * D_stokes * t_ref, "k--", label="6 D t  (Stokes–Einstein)", lw=1.5)
plt.xlabel("Time (s)")
plt.ylabel("MSD (m²)")
plt.title("Mean Square Displacement — timestep comparison")
plt.legend()
plt.grid(True)
plt.savefig(f"figures/msd.png", dpi=800)
plt.tight_layout()
plt.show()

# ── Plot 2: Single trajectory (finest timestep, 1 particle) ───────────────────
fig2d, axes2d = plt.subplots(1, len(delta_ts), figsize=(5*len(delta_ts), 5))
fig3d = plt.figure(figsize=(5*len(delta_ts), 5))

for idx, dt in enumerate(delta_ts):
    N_dt = results[dt]["t"].size
    traj = simulate_particle(N_dt, dt)

    # 2D
    axes2d[idx].plot(traj[:, 0 ], traj[:, 1], lw=1, alpha=1)
    axes2d[idx].set_xlabel("x (m)")
    axes2d[idx].set_ylabel("y (m)")
    axes2d[idx].set_title(f"2D Trajectory  (Δt={dt} s)")
    axes2d[idx].ticklabel_format(style='sci', scilimits=(0, 0))
    axes2d[idx].tick_params(labelsize=7)
    axes2d[idx].grid(True)

    # 3D
    ax3d = fig3d.add_subplot(1, len(delta_ts), idx+1, projection="3d")
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1, alpha=1)
    ax3d.set_xlabel("x", labelpad=10)
    ax3d.set_ylabel("y", labelpad=10)
    ax3d.set_zlabel("z", labelpad=10)
    ax3d.tick_params(axis='both', labelsize=7)
    ax3d.ticklabel_format(style='sci', scilimits=(0, 0))
    ax3d.tick_params(labelsize=7)
    ax3d.set_title(f"3D Trajectory  (Δt={dt} s)")

    fig_single, (ax2d, ax3d_single) = plt.subplots(1, 2, figsize=(12, 5),
                                                     subplot_kw=None)
    fig_single.delaxes(ax3d_single)  # remove the flat placeholder, re-add as 3D
    ax2d.plot(traj[:, 0], traj[:, 1], lw=1, alpha=1)
    ax2d.set_xlabel("x (m)",  labelpad=10)
    ax2d.set_ylabel("y (m)",  labelpad=10)
    ax2d.ticklabel_format(style='sci', scilimits=(0, 0))
    ax2d.tick_params(labelsize=7)
    ax2d.set_title(f"2D Trajectory  (Δt={dt} s)")
    ax2d.grid(True)
    ax3d_s = fig_single.add_subplot(1, 2, 2, projection="3d")
    ax3d_s.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1, alpha=1)
    ax3d_s.set_xlabel("x (m)", labelpad=10)
    ax3d_s.set_ylabel("y (m)", labelpad=10)
    ax3d_s.set_zlabel("z (m)", labelpad=10)
    ax3d_s.ticklabel_format(style='sci', scilimits=(0, 0))
    ax3d_s.tick_params(labelsize=7)
    ax3d_s.set_title(f"3D Trajectory  (Δt={dt} s)")
    fig_single.tight_layout()
    fig_single.savefig(f"figures/trajectory_dt{dt}.png", dpi=800)
    plt.close(fig_single)

fig2d.tight_layout()
fig2d.show()
fig3d.tight_layout()
plt.show() 

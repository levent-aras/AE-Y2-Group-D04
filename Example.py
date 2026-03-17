import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# ============================================================
# Brownian motion of soot particles by Langevin dynamics
# AE2224-I project
# ============================================================

# ----------------------------
# Physical constants
# ----------------------------
R = 8.314  # J mol^-1 K^-1
N_A_TRUE = 6.02214076e23  # mol^-1
k_B = R / N_A_TRUE  # Boltzmann constant, J K^-1

# Soot density
RHO_SOOT = 1800.0  # kg m^-3  (1.8 g/cm^3)

# Dynamic viscosities from the assignment
MU_WATER_300K = 0.853e-3   # Pa s
MU_AIR_300K = 1.85e-5      # Pa s
MU_AIR_1800K = 5.87e-5     # Pa s

# Plot style
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11


def particle_mass(dp_m, rho=RHO_SOOT):
    """
    Mass of a spherical particle.
    dp_m : particle diameter in meters
    """
    volume = (np.pi / 6.0) * dp_m**3
    return rho * volume


def friction_coefficient_stokes(mu, dp_m):
    """
    Stokes drag friction coefficient:
    f = 3*pi*mu*dp
    """
    return 3.0 * np.pi * mu * dp_m


def diffusion_stokes_einstein(T, mu, dp_m):
    """
    Stokes-Einstein diffusion coefficient:
    D = R*T / (f*N_A)
    """
    f = friction_coefficient_stokes(mu, dp_m)
    return R * T / (f * N_A_TRUE)


def simulate_brownian_motion(
    dp_nm,
    T,
    mu,
    dt,
    total_time,
    n_particles=2000,
    seed=42
):
    """
    Simulate Brownian motion using the exact underdamped Langevin update
    for one free particle in 3D.

    We evolve an ensemble of particles to obtain smooth MSD curves.
    One representative particle (index 0) is used for XY and XYZ trajectory plots.

    Returns a dictionary with time, trajectories, MSDs, D estimates, etc.
    """
    rng = np.random.default_rng(seed)

    dp_m = dp_nm * 1e-9
    m = particle_mass(dp_m)
    f = friction_coefficient_stokes(mu, dp_m)
    beta = f / m  # damping rate, 1/s

    n_steps = int(np.round(total_time / dt))
    t = np.arange(n_steps + 1) * dt

    # Arrays: shape = (time, particle, xyz)
    x = np.zeros((n_steps + 1, n_particles, 3), dtype=float)
    v = np.zeros((n_steps + 1, n_particles, 3), dtype=float)

    # Stationary initial velocity distribution (Maxwell-Boltzmann per component)
    sigma_v0 = np.sqrt(k_B * T / m)
    v[0] = rng.normal(0.0, sigma_v0, size=(n_particles, 3))

    # Exact coefficients for one timestep
    c = np.exp(-beta * dt)
    a = (1.0 - c) / beta

    # Velocity noise variance
    G = np.sqrt((k_B * T / m) * (1.0 - c**2))

    # Position-velocity covariance
    cov_rv = (k_B * T / m) * (1.0 - c)**2 / beta

    # Position noise variance
    var_r = (2.0 * k_B * T / m / beta**2) * (
        beta * dt - 1.5 + 2.0 * c - 0.5 * c**2
    )

    # Correlated Gaussian construction:
    # v_{n+1} = c v_n + G Y1
    # x_{n+1} = x_n + a v_n + H Y1 + I Y2
    H = cov_rv / G if G > 0 else 0.0
    I_sq = max(var_r - H**2, 0.0)
    I = np.sqrt(I_sq)

    # Time integration
    for n in range(n_steps):
        Y1 = rng.normal(size=(n_particles, 3))
        Y2 = rng.normal(size=(n_particles, 3))

        v[n + 1] = c * v[n] + G * Y1
        x[n + 1] = x[n] + a * v[n] + H * Y1 + I * Y2

    # Displacements relative to initial position
    disp = x - x[0]

    # Ensemble-averaged MSD per coordinate
    msd_x = np.mean(disp[:, :, 0]**2, axis=1)
    msd_y = np.mean(disp[:, :, 1]**2, axis=1)
    msd_z = np.mean(disp[:, :, 2]**2, axis=1)
    msd_total = msd_x + msd_y + msd_z

    # Diffusion coefficient from slope of total MSD:
    # <r^2> = 6 D t  => slope = 6D
    slope_total, intercept_total = np.polyfit(t[1:], msd_total[1:], 1)
    D_est = slope_total / 6.0

    # Per-axis diffusion estimates
    slope_x, _ = np.polyfit(t[1:], msd_x[1:], 1)
    slope_y, _ = np.polyfit(t[1:], msd_y[1:], 1)
    slope_z, _ = np.polyfit(t[1:], msd_z[1:], 1)

    D_x = slope_x / 2.0
    D_y = slope_y / 2.0
    D_z = slope_z / 2.0

    # Stokes-Einstein theoretical diffusion coefficient
    D_theory = diffusion_stokes_einstein(T, mu, dp_m)

    # Avogadro number estimate from D_est and Stokes-Einstein rearranged
    # N_A = R T / (f D)
    N_A_est = R * T / (f * D_est)

    return {
        "t": t,
        "x": x,
        "v": v,
        "msd_x": msd_x,
        "msd_y": msd_y,
        "msd_z": msd_z,
        "msd_total": msd_total,
        "D_est": D_est,
        "D_theory": D_theory,
        "D_x": D_x,
        "D_y": D_y,
        "D_z": D_z,
        "N_A_est": N_A_est,
        "dp_nm": dp_nm,
        "dp_m": dp_m,
        "T": T,
        "mu": mu,
        "dt": dt,
        "total_time": total_time,
        "n_particles": n_particles,
        "m": m,
        "f": f,
        "beta": beta,
        "G": G,
        "H": H,
        "I": I,
    }


def print_case_summary(title, result):
    """
    Print a clean summary of one simulation case.
    """
    err_D = 100.0 * abs(result["D_est"] - result["D_theory"]) / result["D_theory"]
    err_NA = 100.0 * abs(result["N_A_est"] - N_A_TRUE) / N_A_TRUE

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Particle diameter              : {result['dp_nm']:.1f} nm")
    print(f"Temperature                    : {result['T']:.1f} K")
    print(f"Viscosity                      : {result['mu']:.4e} Pa s")
    print(f"Timestep                       : {result['dt']:.1f} s")
    print(f"Total simulation time          : {result['total_time']:.1f} s")
    print(f"Number of particles (ensemble) : {result['n_particles']}")
    print()
    print(f"Estimated D from MSD           : {result['D_est']:.6e} m^2/s")
    print(f"Theoretical D (Stokes-Einstein): {result['D_theory']:.6e} m^2/s")
    print(f"Relative error in D            : {err_D:.3f} %")
    print()
    print(f"D_x                            : {result['D_x']:.6e} m^2/s")
    print(f"D_y                            : {result['D_y']:.6e} m^2/s")
    print(f"D_z                            : {result['D_z']:.6e} m^2/s")
    print()
    print(f"Estimated Avogadro number      : {result['N_A_est']:.6e} mol^-1")
    print(f"Accepted Avogadro number       : {N_A_TRUE:.6e} mol^-1")
    print(f"Relative error in N_A          : {err_NA:.3f} %")
    print()


def plot_msd(result, title):
    """
    Plot MSDx, MSDy, MSDz versus time.
    """
    t = result["t"]

    plt.figure()
    plt.plot(t, result["msd_x"], label=r'$\langle \Delta x^2 \rangle$')
    plt.plot(t, result["msd_y"], label=r'$\langle \Delta y^2 \rangle$')
    plt.plot(t, result["msd_z"], label=r'$\langle \Delta z^2 \rangle$')
    plt.xlabel("Time [s]")
    plt.ylabel("Mean square displacement [m$^2$]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_xy_trajectory(result, title, particle_index=0):
    """
    Plot XY trajectory of one representative particle.
    """
    traj = result["x"][:, particle_index, :]

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()


def plot_xyz_trajectory(result, title, particle_index=0):
    """
    Plot XYZ trajectory of one representative particle.
    """
    traj = result["x"][:, particle_index, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    plt.tight_layout()


def plot_dt_sweep(results, title):
    """
    Plot estimated D and estimated N_A versus timestep for comparison.
    """
    dts = [r["dt"] for r in results]
    D_est = [r["D_est"] for r in results]
    D_theory = [r["D_theory"] for r in results]
    N_A_est = [r["N_A_est"] for r in results]

    plt.figure()
    plt.plot(dts, D_est, "o-", label="Estimated D")
    plt.plot(dts, D_theory, "s--", label="Theoretical D")
    plt.xlabel("Timestep [s]")
    plt.ylabel("Diffusion coefficient [m$^2$/s]")
    plt.title(title + " - Diffusion coefficient")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(dts, N_A_est, "o-", label=r"Estimated $N_A$")
    plt.axhline(N_A_TRUE, linestyle="--", label=r"Accepted $N_A$")
    plt.xlabel("Timestep [s]")
    plt.ylabel(r"Avogadro number [mol$^{-1}$]")
    plt.title(title + " - Avogadro number")
    plt.legend()
    plt.tight_layout()


def plot_task_iv(dp_list_nm, D_est_list, D_theory_list):
    """
    Plot diffusion coefficient as a function of particle diameter for Task IV.
    """
    plt.figure()
    plt.plot(dp_list_nm, D_est_list, "o-", label="Estimated D from MSD")
    plt.plot(dp_list_nm, D_theory_list, "s--", label="Stokes-Einstein D")
    plt.xlabel("Particle diameter [nm]")
    plt.ylabel("Diffusion coefficient [m$^2$/s]")
    plt.title("Task IV: Diffusion coefficient vs particle diameter")
    plt.legend()
    plt.tight_layout()


def main():
    # ------------------------------------------------------------
    # Adjustable numerical settings
    # ------------------------------------------------------------
    # More particles -> smoother MSD and more accurate D, N_A estimates
    n_particles_main = 2000

    # Long enough for clear linear MSD behavior
    total_time_main = 5000.0  # s

    # For very small particles in Task IV, shorter run is sufficient
    total_time_task_iv = 500.0  # s

    # ------------------------------------------------------------
    # Task I: Water droplet, T = 300 K, dp = 500 nm
    # A-D for dt = 5 s
    # E repeat for dt = 10, 35, 50 s
    # ------------------------------------------------------------
    dt_values = [5.0, 10.0, 35.0, 50.0]
    task_I_results = []

    for i, dt in enumerate(dt_values):
        res = simulate_brownian_motion(
            dp_nm=500.0,
            T=300.0,
            mu=MU_WATER_300K,
            dt=dt,
            total_time=total_time_main,
            n_particles=n_particles_main,
            seed=100 + i
        )
        task_I_results.append(res)
        print_case_summary(f"Task I - Water, 300 K, dp = 500 nm, dt = {dt:.0f} s", res)

    # Plots required explicitly for dt = 5 s
    res_I_5 = task_I_results[0]
    plot_msd(res_I_5, "Task I.A: MSD vs time (water, 300 K, dp = 500 nm, dt = 5 s)")
    plot_xy_trajectory(res_I_5, "Task I.B: XY trajectory (water, 300 K, dp = 500 nm, dt = 5 s)")
    plot_xyz_trajectory(res_I_5, "Task I.B: XYZ trajectory (water, 300 K, dp = 500 nm, dt = 5 s)")
    plot_dt_sweep(task_I_results, "Task I.E: Water, 300 K, dp = 500 nm")

    # ------------------------------------------------------------
    # Task II: Air, T = 300 K, dp = 500 nm
    # Repeat A-D
    # ------------------------------------------------------------
    res_II = simulate_brownian_motion(
        dp_nm=500.0,
        T=300.0,
        mu=MU_AIR_300K,
        dt=5.0,
        total_time=total_time_main,
        n_particles=n_particles_main,
        seed=200
    )

    print_case_summary("Task II - Air, 300 K, dp = 500 nm, dt = 5 s", res_II)

    plot_msd(res_II, "Task II.A: MSD vs time (air, 300 K, dp = 500 nm, dt = 5 s)")
    plot_xy_trajectory(res_II, "Task II.B: XY trajectory (air, 300 K, dp = 500 nm, dt = 5 s)")
    plot_xyz_trajectory(res_II, "Task II.B: XYZ trajectory (air, 300 K, dp = 500 nm, dt = 5 s)")

    # ------------------------------------------------------------
    # Task III: Air, T = 1800 K, dp = 500 nm
    # Repeat A-D
    # ------------------------------------------------------------
    res_III = simulate_brownian_motion(
        dp_nm=500.0,
        T=1800.0,
        mu=MU_AIR_1800K,
        dt=5.0,
        total_time=total_time_main,
        n_particles=n_particles_main,
        seed=300
    )

    print_case_summary("Task III - Air, 1800 K, dp = 500 nm, dt = 5 s", res_III)

    plot_msd(res_III, "Task III.A: MSD vs time (air, 1800 K, dp = 500 nm, dt = 5 s)")
    plot_xy_trajectory(res_III, "Task III.B: XY trajectory (air, 1800 K, dp = 500 nm, dt = 5 s)")
    plot_xyz_trajectory(res_III, "Task III.B: XYZ trajectory (air, 1800 K, dp = 500 nm, dt = 5 s)")

    # ------------------------------------------------------------
    # Task IV: Air, T = 1800 K, dp = 20, 40, 60, 80 nm
    # Estimate D and compare to Stokes-Einstein
    # ------------------------------------------------------------
    dp_list_nm = [20.0, 40.0, 60.0, 80.0]
    D_est_list = []
    D_theory_list = []

    for i, dp_nm in enumerate(dp_list_nm):
        res = simulate_brownian_motion(
            dp_nm=dp_nm,
            T=1800.0,
            mu=MU_AIR_1800K,
            dt=5.0,
            total_time=total_time_task_iv,
            n_particles=n_particles_main,
            seed=400 + i
        )

        print_case_summary(f"Task IV - Air, 1800 K, dp = {dp_nm:.0f} nm, dt = 5 s", res)

        D_est_list.append(res["D_est"])
        D_theory_list.append(res["D_theory"])

    plot_task_iv(dp_list_nm, D_est_list, D_theory_list)

    # ------------------------------------------------------------
    # Final console comparison notes
    # ------------------------------------------------------------
    print("=" * 70)
    print("Interpretation hints")
    print("=" * 70)
    print("1. For ideal Brownian diffusion, <Δx^2>, <Δy^2>, <Δz^2> should grow")
    print("   approximately linearly with time.")
    print("2. Lower viscosity and/or higher temperature increase D.")
    print("3. Therefore, D in air is much larger than in water.")
    print("4. At 1800 K, D is larger than at 300 K.")
    print("5. In Task IV, D should decrease approximately as 1/d_p.")
    print("6. Very large timesteps can reduce accuracy slightly, although the")
    print("   exact Langevin update used here is much more stable than a simple")
    print("   Euler-Maruyama approach.")

    plt.show()


if __name__ == "__main__":
    main()
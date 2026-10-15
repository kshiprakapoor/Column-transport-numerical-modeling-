# %% 
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FPLC V4
# 1D ADVECTION-DISPERSION MODEL
#
# Governing equation:
#
# dC/dt + u*dC/dz = Dax*d2C/dz2
#
# Represents transport of an injected sample band through
# the packed SEC column.
# ============================================================


# ------------------------------------------------------------
# COLUMN
# ------------------------------------------------------------
L = 0.20                 # column length [m]
Nz = 201                 # number of spatial grid points

z = np.linspace(0, L, Nz)
dz = z[1] - z[0]


# ------------------------------------------------------------
# FLOW PARAMETERS
# ------------------------------------------------------------
D_column = 0.016
A = np.pi * D_column**2 / 4

epsilon = 0.40

Q_L_min = 0.005
Q = Q_L_min * 1e-3 / 60.0

u_superficial = Q / A

# Actual velocity through void space
u = u_superficial / epsilon


# ------------------------------------------------------------
# AXIAL DISPERSION
#
# Representative teaching value.
# Not yet claimed as experimentally measured.
# ------------------------------------------------------------
Dax = 2.0e-6             # [m^2/s]


# ------------------------------------------------------------
# TIME GRID
# ------------------------------------------------------------
dt = 0.10                # time step [s]

t_end = 400.0

Nt = int(t_end / dt) + 1

time = np.linspace(0, t_end, Nt)


# ------------------------------------------------------------
# STABILITY NUMBERS
#
# Advection CFL:
#     CFL = u*dt/dz
#
# Diffusion number:
#     Fo = D*dt/dz^2
#
# Explicit scheme generally requires:
#     CFL <= 1
#     Fo <= 0.5
# ------------------------------------------------------------
CFL = u * dt / dz
Fo = Dax * dt / dz**2


print("\n===== FPLC V4: ADVECTION-DISPERSION =====")

print(f"Column length                 = {L:.3f} m")
print(f"Grid spacing                  = {dz:.6f} m")

print(f"\nSuperficial velocity          = {u_superficial:.6e} m/s")
print(f"Interstitial velocity         = {u:.6e} m/s")

print(f"\nAxial dispersion coefficient  = {Dax:.3e} m^2/s")

print(f"\nTime step                     = {dt:.3f} s")
print(f"CFL number                    = {CFL:.4f}")
print(f"Diffusion number              = {Fo:.4f}")


if CFL > 1:
    print("WARNING: Advection CFL > 1")

if Fo > 0.5:
    print("WARNING: Diffusion stability condition exceeded")


# ------------------------------------------------------------
# CONCENTRATION INITIAL CONDITION
# ------------------------------------------------------------
C = np.zeros(Nz)

# Store outlet chromatogram
C_out = np.zeros(Nt)


# ------------------------------------------------------------
# SAMPLE INJECTION
#
# Inject concentration = 1 for first 10 seconds.
# After that switch inlet back to buffer concentration = 0.
# ------------------------------------------------------------
injection_duration = 10.0


# ------------------------------------------------------------
# SAVE CONCENTRATION PROFILES AT SELECTED TIMES
# ------------------------------------------------------------
snapshot_times = [25, 75, 125, 175, 225, 275]

snapshots = {}


# ============================================================
# NUMERICAL SOLUTION
# ============================================================

for n in range(Nt):

    t = time[n]

    # -----------------------------------------
    # Inlet boundary condition
    # -----------------------------------------
    if t <= injection_duration:
        C[0] = 1.0
    else:
        C[0] = 0.0


    # -----------------------------------------
    # Copy old concentration
    # -----------------------------------------
    C_old = C.copy()


    # -----------------------------------------
    # Interior nodes
    #
    # First-order upwind advection
    # Central-difference diffusion
    # -----------------------------------------
    for i in range(1, Nz - 1):

        advection = (
            -u
            * (C_old[i] - C_old[i-1])
            / dz
        )

        dispersion = (
            Dax
            * (C_old[i+1]
               - 2*C_old[i]
               + C_old[i-1])
            / dz**2
        )

        C[i] = (
            C_old[i]
            + dt * (advection + dispersion)
        )


    # -----------------------------------------
    # Outlet boundary:
    # zero concentration gradient
    # -----------------------------------------
    C[-1] = C[-2]


    # -----------------------------------------
    # Store outlet concentration
    # -----------------------------------------
    C_out[n] = C[-1]


    # -----------------------------------------
    # Save selected concentration profiles
    # -----------------------------------------
    for ts in snapshot_times:

        if abs(t - ts) < dt/2:

            snapshots[ts] = C.copy()


# ============================================================
# FIND OUTLET PEAK
# ============================================================

peak_index = np.argmax(C_out)

peak_time = time[peak_index]
peak_concentration = C_out[peak_index]


print(f"\nInjection duration            = {injection_duration:.1f} s")

print(f"\nPredicted advective travel time = {L/u:.2f} s")

print(f"Outlet peak time                = {peak_time:.2f} s")
print(f"Outlet peak concentration       = {peak_concentration:.4f}")


# ============================================================
# PLOT 1
# Concentration moving through column
# ============================================================

plt.figure(figsize=(8, 5))

for ts in snapshot_times:

    if ts in snapshots:

        plt.plot(
            z,
            snapshots[ts],
            label=f"{ts} s"
        )

plt.xlabel("Axial Position z [m]")
plt.ylabel("Normalized Concentration")

plt.title(
    "FPLC Sample Transport Through Packed Column"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2
# Simulated outlet chromatogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    time,
    C_out
)

plt.axvline(
    peak_time,
    linestyle="--",
    label=f"Peak = {peak_time:.1f} s"
)

plt.xlabel("Time [s]")
plt.ylabel("Normalized Outlet Concentration")

plt.title(
    "Simulated FPLC Chromatogram"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# %%

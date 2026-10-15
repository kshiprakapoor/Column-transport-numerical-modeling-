# %% 
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FPLC V5
# TWO-SPECIES SIZE-EXCLUSION CHROMATOGRAPHY MODEL
#
# Species:
#   1. EV-like large particles
#   2. Albumin/protein-like smaller solute
#
# SEC retention:
#
#       Ve = V0 + Kav * Vi
#
# Large excluded species -> Kav ~ 0 -> earlier elution
# Smaller species        -> Kav > 0 -> later elution
#
# NOTE:
# Parameter values below are illustrative teaching values.
# They are NOT yet fitted to experimental qEV10 data.
# ============================================================


# ------------------------------------------------------------
# COLUMN GEOMETRY
# ------------------------------------------------------------

L = 0.20
D_column = 0.016

A = np.pi * D_column**2 / 4

V_column = A * L


# ------------------------------------------------------------
# BED VOLUMES
# ------------------------------------------------------------

epsilon_b = 0.40       # interparticle bed void fraction

V0 = epsilon_b * V_column

# Illustrative internal bead pore volume
Vi = 0.35 * V_column


# ------------------------------------------------------------
# FLOW RATE
# ------------------------------------------------------------

Q_L_min = 0.005
Q = Q_L_min * 1e-3 / 60.0


# ------------------------------------------------------------
# SPECIES ACCESSIBILITY
# ------------------------------------------------------------

# EV-like particle:
# almost completely excluded from internal pores

Kav_EV = 0.05

# Albumin-like soluble protein:
# substantial access to internal pores

Kav_albumin = 0.65


# ------------------------------------------------------------
# RETENTION VOLUMES
# ------------------------------------------------------------

Ve_EV = V0 + Kav_EV * Vi

Ve_albumin = V0 + Kav_albumin * Vi


# ------------------------------------------------------------
# EXPECTED RETENTION TIMES
#
# t = Ve / Q
# ------------------------------------------------------------

tR_EV = Ve_EV / Q
tR_albumin = Ve_albumin / Q


# ------------------------------------------------------------
# EFFECTIVE TRANSPORT VELOCITIES
#
# u = L / tR
# ------------------------------------------------------------

u_EV = L / tR_EV
u_albumin = L / tR_albumin


# ------------------------------------------------------------
# AXIAL DISPERSION
#
# Illustrative values
# ------------------------------------------------------------

Dax_EV = 1.5e-6
Dax_albumin = 2.0e-6


# ------------------------------------------------------------
# SPATIAL GRID
# ------------------------------------------------------------

Nz = 201

z = np.linspace(0, L, Nz)

dz = z[1] - z[0]


# ------------------------------------------------------------
# TIME GRID
# ------------------------------------------------------------

dt = 0.05

t_end = 600.0

time = np.arange(0, t_end + dt, dt)

Nt = len(time)


# ------------------------------------------------------------
# STABILITY CHECKS
# ------------------------------------------------------------

CFL_EV = u_EV * dt / dz
CFL_albumin = u_albumin * dt / dz

Fo_EV = Dax_EV * dt / dz**2
Fo_albumin = Dax_albumin * dt / dz**2


# ------------------------------------------------------------
# INITIAL CONDITIONS
# ------------------------------------------------------------

C_EV = np.zeros(Nz)

C_albumin = np.zeros(Nz)


# Outlet chromatograms

Cout_EV = np.zeros(Nt)

Cout_albumin = np.zeros(Nt)


# ------------------------------------------------------------
# INJECTION
# ------------------------------------------------------------

injection_duration = 10.0


# ============================================================
# TRANSPORT FUNCTION
# ============================================================

def update_species(C, u, Dax, inlet_value):

    C_old = C.copy()

    # inlet concentration
    C[0] = inlet_value

    for i in range(1, Nz - 1):

        # upwind advection
        advective_term = (
            -u
            * (C_old[i] - C_old[i-1])
            / dz
        )

        # axial dispersion
        dispersive_term = (
            Dax
            * (
                C_old[i+1]
                - 2*C_old[i]
                + C_old[i-1]
            )
            / dz**2
        )

        C[i] = (
            C_old[i]
            + dt
            * (
                advective_term
                + dispersive_term
            )
        )

    # zero-gradient outlet
    C[-1] = C[-2]

    return C


# ============================================================
# TIME INTEGRATION
# ============================================================

for n, t in enumerate(time):

    # finite injection pulse

    if t <= injection_duration:
        inlet = 1.0
    else:
        inlet = 0.0


    # EV transport

    C_EV = update_species(
        C_EV,
        u_EV,
        Dax_EV,
        inlet
    )


    # Albumin transport

    C_albumin = update_species(
        C_albumin,
        u_albumin,
        Dax_albumin,
        inlet
    )


    # outlet detector

    Cout_EV[n] = C_EV[-1]

    Cout_albumin[n] = C_albumin[-1]


# ============================================================
# PEAK ANALYSIS
# ============================================================

idx_EV = np.argmax(Cout_EV)
idx_albumin = np.argmax(Cout_albumin)

peak_EV = time[idx_EV]
peak_albumin = time[idx_albumin]

separation_time = peak_albumin - peak_EV


# ============================================================
# RESULTS
# ============================================================

print("\n===== FPLC V5: TWO-SPECIES SEC MODEL =====")

print(f"\nColumn volume                = {V_column*1e6:.2f} mL")

print(f"Void volume V0              = {V0*1e6:.2f} mL")
print(f"Internal pore volume Vi     = {Vi*1e6:.2f} mL")

print("\n--- EV-like species ---")

print(f"Kav                          = {Kav_EV:.2f}")
print(f"Retention volume            = {Ve_EV*1e6:.2f} mL")
print(f"Expected retention time     = {tR_EV:.1f} s")
print(f"Effective velocity          = {u_EV:.6e} m/s")
print(f"Numerical peak time         = {peak_EV:.1f} s")


print("\n--- Albumin-like species ---")

print(f"Kav                          = {Kav_albumin:.2f}")
print(f"Retention volume            = {Ve_albumin*1e6:.2f} mL")
print(f"Expected retention time     = {tR_albumin:.1f} s")
print(f"Effective velocity          = {u_albumin:.6e} m/s")
print(f"Numerical peak time         = {peak_albumin:.1f} s")


print("\n--- Separation ---")

print(f"Peak separation             = {separation_time:.1f} s")

print(f"\nCFL EV                      = {CFL_EV:.4f}")
print(f"CFL albumin                 = {CFL_albumin:.4f}")

print(f"Fo EV                       = {Fo_EV:.4f}")
print(f"Fo albumin                  = {Fo_albumin:.4f}")


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time / 60,
    Cout_EV,
    label="EV-like species"
)

plt.plot(
    time / 60,
    Cout_albumin,
    label="Albumin-like species"
)

plt.axvline(
    peak_EV / 60,
    linestyle="--"
)

plt.axvline(
    peak_albumin / 60,
    linestyle="--"
)

plt.xlabel("Time [min]")

plt.ylabel(
    "Normalized Outlet Concentration"
)

plt.title(
    "Simulated Size-Exclusion FPLC Chromatogram"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()
# %%

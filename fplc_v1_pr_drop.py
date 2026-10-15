# %%
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FPLC V1: 1D PACKED-BED PRESSURE DROP
#
# Governing equation:
#
#       dP/dz = -mu*Q/(k*A)
#
# We solve from the outlet back toward the inlet.
# ============================================================

# -----------------------------
# Physical parameters
# -----------------------------
L = 0.20                 # column length [m]
D = 0.016                # column diameter [m]

mu = 1.0e-3              # dynamic viscosity [Pa.s]
k = 2.0e-12              # permeability [m^2]

Q_L_min = 0.005          # flow rate [L/min]

P_out = 0.0              # outlet gauge pressure [Pa]

dz = 0.005               # spatial step [m]


# -----------------------------
# Unit conversion
# -----------------------------
Q = Q_L_min * 1e-3 / 60.0    # L/min -> m^3/s


# -----------------------------
# Geometry
# -----------------------------
A = np.pi * D**2 / 4.0


# -----------------------------
# Grid
# -----------------------------
z = np.arange(0.0, L + dz, dz)

N = len(z)

P = np.zeros(N)

# We define:
#
# z = 0      -> inlet
# z = L      -> outlet
#
# Known boundary condition:
# P(L) = P_out

P[-1] = P_out


# -----------------------------
# Constant pressure gradient
# -----------------------------
dP_dz = -mu * Q / (k * A)


# -----------------------------
# Numerical solution
#
# March backward from outlet:
#
# P[i] = P[i+1] - (dP/dz)*dz
# -----------------------------
for i in range(N - 2, -1, -1):
    P[i] = P[i + 1] - dP_dz * dz


# -----------------------------
# Analytical solution
# -----------------------------
P_analytical = P_out + (mu * Q / (k * A)) * (L - z)


# -----------------------------
# Verification
# -----------------------------
max_error = np.max(np.abs(P - P_analytical))

deltaP = P[0] - P[-1]


# -----------------------------
# Derived quantities
# -----------------------------
superficial_velocity = Q / A

residence_time = L / superficial_velocity


# -----------------------------
# Print results
# -----------------------------
print("\n===== FPLC V1: PACKED-BED PRESSURE DROP =====")

print(f"Column length                = {L:.3f} m")
print(f"Column diameter              = {D:.4f} m")
print(f"Column area                  = {A:.6e} m^2")

print(f"\nFlow rate                    = {Q_L_min:.4f} L/min")
print(f"Flow rate                    = {Q:.6e} m^3/s")

print(f"\nDynamic viscosity            = {mu:.3e} Pa.s")
print(f"Permeability                 = {k:.3e} m^2")

print(f"\nSuperficial velocity         = {superficial_velocity:.6e} m/s")
print(f"Estimated residence time     = {residence_time:.2f} s")

print(f"\nPressure gradient            = {abs(dP_dz):.2f} Pa/m")

print(f"Inlet pressure               = {P[0]:.2f} Pa")
print(f"Outlet pressure              = {P[-1]:.2f} Pa")
print(f"Total pressure drop          = {deltaP:.2f} Pa")

print(f"\nMaximum numerical error      = {max_error:.6e} Pa")


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

plt.plot(z, P / 1000.0, 'o-', label="Numerical")
plt.plot(z, P_analytical / 1000.0, '--', label="Analytical")

plt.xlabel("Axial Position z [m]")
plt.ylabel("Gauge Pressure [kPa]")

plt.title("FPLC Packed-Bed Pressure Profile")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# %%

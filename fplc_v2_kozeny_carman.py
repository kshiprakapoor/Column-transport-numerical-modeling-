# %%
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FPLC V2
# Packed-bed pressure drop using Kozeny-Carman permeability
# ============================================================

# -----------------------------
# Geometry
# -----------------------------
L = 0.20                 # column length [m]
D = 0.016                # column diameter [m]
A = np.pi * D**2 / 4.0

# -----------------------------
# Fluid
# -----------------------------
mu = 1.0e-3              # dynamic viscosity [Pa.s]

# -----------------------------
# Packed bed
# -----------------------------
epsilon = 0.40            # bed porosity [-]
dp = 75e-6                # bead diameter [m]

# Kozeny-Carman permeability
k = (epsilon**3 * dp**2) / (180.0 * (1.0 - epsilon)**2)

# -----------------------------
# Flow
# -----------------------------
Q_L_min = 0.005
Q = Q_L_min * 1e-3 / 60.0

u_superficial = Q / A
u_interstitial = u_superficial / epsilon

# -----------------------------
# Pressure model
# -----------------------------
P_out = 0.0
dz = 0.005

z = np.arange(0.0, L + dz, dz)
N = len(z)

P = np.zeros(N)
P[-1] = P_out

dP_dz = -mu * Q / (k * A)

for i in range(N - 2, -1, -1):
    P[i] = P[i + 1] - dP_dz * dz

# analytical solution
P_analytical = P_out + (mu * Q / (k * A)) * (L - z)

max_error = np.max(np.abs(P - P_analytical))
deltaP = P[0] - P[-1]

# -----------------------------
# Residence times
# -----------------------------
t_superficial = L / u_superficial
t_interstitial = L / u_interstitial

# -----------------------------
# Reynolds number based on beads
# -----------------------------
rho = 1000.0

Re_p = rho * u_superficial * dp / mu

# -----------------------------
# Results
# -----------------------------
print("\n===== FPLC V2: KOZENY-CARMAN PACKED BED =====")

print(f"Column length                 = {L:.3f} m")
print(f"Column diameter               = {D:.4f} m")
print(f"Column area                   = {A:.6e} m^2")

print(f"\nBead diameter                 = {dp*1e6:.1f} um")
print(f"Bed porosity                  = {epsilon:.3f}")
print(f"Calculated permeability       = {k:.6e} m^2")

print(f"\nFlow rate                     = {Q_L_min:.4f} L/min")

print(f"Superficial velocity          = {u_superficial:.6e} m/s")
print(f"Interstitial velocity         = {u_interstitial:.6e} m/s")

print(f"\nSuperficial residence time    = {t_superficial:.2f} s")
print(f"Interstitial residence time   = {t_interstitial:.2f} s")

print(f"\nParticle Reynolds number      = {Re_p:.6f}")

print(f"\nPressure gradient             = {abs(dP_dz):.2f} Pa/m")
print(f"Inlet pressure                = {P[0]:.2f} Pa")
print(f"Total pressure drop           = {deltaP:.2f} Pa")

print(f"\nMaximum numerical error       = {max_error:.6e} Pa")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))

plt.plot(z, P/1000, 'o-', label="Numerical")
plt.plot(z, P_analytical/1000, '--', label="Analytical")

plt.xlabel("Axial Position z [m]")
plt.ylabel("Gauge Pressure [kPa]")
plt.title("FPLC Packed-Bed Pressure Profile\nKozeny-Carman Permeability")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
# %%

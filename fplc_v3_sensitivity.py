# %% 
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FPLC V3: SENSITIVITY ANALYSIS
#
# Investigate how:
#   1) bead diameter
#   2) bed porosity
#   3) flow rate
#
# affect packed-bed pressure drop
# ============================================================

# ------------------------------------------------------------
# BASELINE PARAMETERS
# ------------------------------------------------------------
L = 0.20                 # column length [m]
D = 0.016                # column diameter [m]
A = np.pi * D**2 / 4.0

mu = 1.0e-3              # dynamic viscosity [Pa.s]

epsilon_base = 0.40      # baseline porosity [-]
dp_base = 75e-6          # baseline bead diameter [m]
Q_base_L_min = 0.005     # baseline flow rate [L/min]


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
def permeability_kozeny_carman(dp, epsilon):
    """
    Kozeny-Carman permeability:
    
    k = epsilon^3 * dp^2 /
        [180 * (1-epsilon)^2]
    """
    return (epsilon**3 * dp**2) / (180.0 * (1.0 - epsilon)**2)


def pressure_drop(dp, epsilon, Q_L_min):
    """
    Compute total Darcy pressure drop across the packed bed.
    """

    # convert flow rate
    Q = Q_L_min * 1e-3 / 60.0

    # permeability
    k = permeability_kozeny_carman(dp, epsilon)

    # Darcy pressure drop
    deltaP = mu * Q * L / (k * A)

    return deltaP


# ============================================================
# BASELINE
# ============================================================

deltaP_base = pressure_drop(
    dp_base,
    epsilon_base,
    Q_base_L_min
)

print("\n===== FPLC V3: SENSITIVITY ANALYSIS =====")

print(f"Baseline bead diameter     = {dp_base*1e6:.1f} um")
print(f"Baseline porosity          = {epsilon_base:.3f}")
print(f"Baseline flow rate         = {Q_base_L_min:.4f} L/min")

print(f"\nBaseline pressure drop     = {deltaP_base/1000:.3f} kPa")


# ============================================================
# 1. BEAD DIAMETER SWEEP
# ============================================================

dp_values_um = np.array(
    [40, 50, 60, 75, 90, 110, 130]
)

dp_values = dp_values_um * 1e-6

deltaP_dp = []

for dp in dp_values:

    dP = pressure_drop(
        dp,
        epsilon_base,
        Q_base_L_min
    )

    deltaP_dp.append(dP)

deltaP_dp = np.array(deltaP_dp)


# ============================================================
# 2. POROSITY SWEEP
# ============================================================

epsilon_values = np.array(
    [0.30, 0.33, 0.36, 0.40, 0.44, 0.48, 0.52]
)

deltaP_epsilon = []

for epsilon in epsilon_values:

    dP = pressure_drop(
        dp_base,
        epsilon,
        Q_base_L_min
    )

    deltaP_epsilon.append(dP)

deltaP_epsilon = np.array(deltaP_epsilon)


# ============================================================
# 3. FLOW RATE SWEEP
# ============================================================

Q_values_L_min = np.array(
    [0.001, 0.002, 0.003, 0.005, 0.0075, 0.010, 0.015]
)

deltaP_Q = []

for Q_L_min in Q_values_L_min:

    dP = pressure_drop(
        dp_base,
        epsilon_base,
        Q_L_min
    )

    deltaP_Q.append(dP)

deltaP_Q = np.array(deltaP_Q)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n--- Bead Diameter Sensitivity ---")

for dp_um, dP in zip(dp_values_um, deltaP_dp):
    print(
        f"dp = {dp_um:6.1f} um"
        f"   -> DeltaP = {dP/1000:8.3f} kPa"
    )


print("\n--- Porosity Sensitivity ---")

for epsilon, dP in zip(epsilon_values, deltaP_epsilon):
    print(
        f"epsilon = {epsilon:.2f}"
        f"   -> DeltaP = {dP/1000:8.3f} kPa"
    )


print("\n--- Flow Rate Sensitivity ---")

for Q_L_min, dP in zip(Q_values_L_min, deltaP_Q):
    print(
        f"Q = {Q_L_min:7.4f} L/min"
        f"   -> DeltaP = {dP/1000:8.3f} kPa"
    )


# ============================================================
# PLOT 1: BEAD DIAMETER
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    dp_values_um,
    deltaP_dp / 1000,
    'o-'
)

plt.xlabel("Bead Diameter [um]")
plt.ylabel("Pressure Drop [kPa]")

plt.title(
    "FPLC Pressure Drop Sensitivity\n"
    "Effect of Bead Diameter"
)

plt.grid(True)
plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2: POROSITY
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    epsilon_values,
    deltaP_epsilon / 1000,
    'o-'
)

plt.xlabel("Bed Porosity [-]")
plt.ylabel("Pressure Drop [kPa]")

plt.title(
    "FPLC Pressure Drop Sensitivity\n"
    "Effect of Bed Porosity"
)

plt.grid(True)
plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3: FLOW RATE
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    Q_values_L_min,
    deltaP_Q / 1000,
    'o-'
)

plt.xlabel("Flow Rate [L/min]")
plt.ylabel("Pressure Drop [kPa]")

plt.title(
    "FPLC Pressure Drop Sensitivity\n"
    "Effect of Flow Rate"
)

plt.grid(True)
plt.tight_layout()

plt.show()
# %%

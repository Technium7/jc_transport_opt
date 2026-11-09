"""Physical scales for the Y-junction transport.

Positions in the junction geometry and on the transport path are in
dimensionless trap heights. One trap height is ``L_SCALE`` metres.
"""

import numpy as np
import scipy.constants as ct

# Metres per trap height. The ion's default height is one trap height, 50 um.
L_SCALE = 50e-6
ION_HEIGHT_M = 50e-6

# The electrode system is built at rf = 1. This amplitude converts that
# dimensionless pseudopotential into a physical curvature.
V_RF = 20.0
OMEGA_RF = 2.0 * np.pi * 40e6

F_AXIAL = 1e6
F_RADIAL = 3e6

# Ion integration step. Voltage updates are an integer number of these steps.
SIM_DT = 1e-9

MASS = 40 * ct.atomic_mass
CHARGE = ct.elementary_charge
HBAR = ct.hbar

# Default ion height. Locked on the straight arms and at the crossing.
ION_HEIGHT = ION_HEIGHT_M / L_SCALE

# Collection and optimization stay inside these limits.
# Bend controls may leave 50 um by at most 10 um, and the crossing lies
# at most 10 um from the origin along the +120 degree line.
HEIGHT_CAP_M = 10e-6
CROSS_CAP_M = 10e-6

# Fixed electrode half-gap and RF rail width, in trap heights.
RF_SEP = 1.0
RF_WIDTH = 1.5

RF_SCALE = np.sqrt(CHARGE / MASS) / (2.0 * L_SCALE * OMEGA_RF)
OMEGA_AX = 2.0 * np.pi * F_AXIAL
OMEGA_RAD = 2.0 * np.pi * F_RADIAL

# Dimensionless curvature (electrode units) that produces f_axial / f_radial
# once positions are converted with L_SCALE.
KAPPA_AX = OMEGA_AX ** 2 * MASS * L_SCALE ** 2 / CHARGE
KAPPA_RAD = OMEGA_RAD ** 2 * MASS * L_SCALE ** 2 / CHARGE

# Each target DC well is held for 100 ns. The stray field is the residual of a
# real voltage solution: 100 V/m is the standard deviation of its maximum on
# each path-frame axis. Twenty independent draws per design give the stochastic
# network a mean and a variance to fit. A quiet run sets the amplitude to 0.
VOLTAGE_UPDATE_S = 100e-9
NOISE_STD_VM = 100.0
NOISE_SEEDS = tuple(range(20))
TRANSPORT_SPEED_M_S = 7.5

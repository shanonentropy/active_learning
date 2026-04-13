import qutip as qt 
import numpy as np

##################   Define basis functions for the 7 level basis
# gs - ground electronic state, 0- m=0, m1- m=-1, p1- m=+1
# es - excited electronic state, 0- m=0, m1- m=-1, p1- m=+1
# shelf - singlet shelving state

gs_0 = qt.basis(7,0)
gs_m1 = qt.basis(7,1)
gs_p1 = qt.basis(7,2)
es_0 = qt.basis(7,3)
es_m1 = qt.basis(7,4)
es_p1 = qt.basis(7,5)
shelf_state = qt.basis(7,6)


'''When no applied magnetic field, all spin states are degenerate

The NV center optical transition energy: 637.1 nm ~ 1.945 eV ~ 470.8 THz = 4.708 x 10^(8) MHz
We have placed the shelving state halfway between the ground and excited states, which is technically not correct but a good enough approximation
factor of 2*pi to make this into an angular frequency '''
H_0 = 2 * np.pi * (4.708E8*es_p1*es_p1.dag()
                    + 4.708E8*es_0*es_0.dag()
                    + 4.708E8*es_m1*es_m1.dag()
                    + 0.5*4.708E8*shelf_state*shelf_state.dag())

''' Define the density matrix, assuming initial density is thermal (1/3 in each of the three ground states) '''

rho_0 = (1/3)*(gs_0*gs_0.dag() + gs_m1*gs_m1.dag() + gs_p1*gs_p1.dag())


''' Define the initial transition rates for the different states (denominator is in microseconds) ie rates are in MHz '''
# where are these numbers from - generally, Lucio Robledo's work - https://iopscience.iop.org/article/10.1088/1367-2630/13/2/025013/pdf

rate_laser = 1/0.01 #rate of excitation  
rate_eg = 1/0.012 #rate of relaxation from the ES to GS (give off a photon)
rate_isc = 1/0.045 #rate of relaxing from the ES to the shelve state
rate_sg = 1/0.300 #rate of relaxing from the shelve state to the ground state
rate_T1 = 1E-99 # rate of T1 process happening in the GS/ES prob= e^-t/T1. Usually T1 is in ms, we just make T1 rate very small here


''' Define the collapse operators for the Lindblad master equation '''
dim = 7

# Laser Pumping (rename variables to cop_ab where a=source index, b=dest index)
cop_03 = np.sqrt(rate_laser)*es_0*gs_0.dag() # 0 -> 3 (gs_0 -> es_0)
cop_14 = np.sqrt(rate_laser)*es_m1*gs_m1.dag() # 1 -> 4 (gs_m1 -> es_m1)
cop_25 = np.sqrt(rate_laser)*es_p1*gs_p1.dag() # 2 -> 5 (gs_p1 -> es_p1)

# T1 processes (ground-state flips) - names corrected to match operator semantics
cop_01 = np.sqrt(rate_T1)*gs_m1*gs_0.dag()  # 0 -> 1
cop_12 = np.sqrt(rate_T1)*gs_p1*gs_m1.dag() # 1 -> 2
cop_20 = np.sqrt(rate_T1)*gs_0*gs_p1.dag()  # 2 -> 0

# provide dag versions to make the spin flipping bidirectional
cop_10 = cop_01.dag()
cop_21 = cop_12.dag()
cop_02 = cop_20.dag()

# Spontaneous emission (excited -> ground) - corrected naming
cop_30 = np.sqrt(rate_eg)*gs_0*es_0.dag() # 3 -> 0
cop_41 = np.sqrt(rate_eg)*gs_m1*es_m1.dag() # 4 -> 1
cop_52 = np.sqrt(rate_eg)*gs_p1*es_p1.dag() # 5 -> 2

# ISC (excited -> shelf) - corrected naming and keep expressions
cop_36 = np.sqrt(rate_isc/10)*shelf_state*es_0.dag() # 3 -> 6 (es_0 -> shelf)
cop_46 = np.sqrt(rate_isc)*shelf_state*es_m1.dag()     # 4 -> 6
cop_56 = np.sqrt(rate_isc)*shelf_state*es_p1.dag()     # 5 -> 6

cop_60 = np.sqrt(rate_sg)*gs_0*shelf_state.dag() # 6 -> 0
cop_61 = np.sqrt(rate_sg)*gs_m1*shelf_state.dag()
cop_62 = np.sqrt(rate_sg)*gs_p1*shelf_state.dag()

cops = [cop_03, cop_14, cop_25, cop_01, cop_12, cop_20, cop_10, cop_21, cop_02, cop_30, cop_41, cop_52, cop_36, cop_46, cop_56, cop_60, cop_61, cop_62]


'''define the Sz operator for the NV center'''

Sz = gs_0*gs_0.dag() - gs_p1*gs_p1.dag() + es_0*es_0.dag() - es_p1*es_p1.dag()


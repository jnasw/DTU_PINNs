import pandas as pd
import yaml
import numpy as np

############################################################################################################
"""generator parameters"""
# parameters for the generator from pinnsim paper and p.179 table 7.3
parameters = [ # H(secs), Xd(pu), X_d_dash(pu), Xq(pu), X_q_dash(pu), T_d_dash(sec), T_q_dash(sec) , D(pu), Rs(pu), P_m(pu), E_fd(pu)
    [23.64, 0.146, 0.0608, 0.0969, 0.0969, 8.96, 0.31, 2.364, 0, 0.71, 1.08],
    [6.4, 0.8958, 0.1198, 0.8645, 0.1969, 6.0, 0.535, 1.28, 0, 1.612, 1.32],
    [3.01, 1.3125, 0.1813, 1.2578, 0.25, 5.89, 0.6, 0.903, 0, 0.859, 1.04],
    [5.148, 0.8979, 0.2995, 0.646, 0.646, 7.4, 0.1, 2, 0, 2.32, 1], # from dynamics bus 1
    [6.54, 1.05, 0.185, 0.98, 0.36, 6.1, 0.4, 2, 0, -0.942, 1], # bus 3
    [5.06, 1.25, 0.232, 1.22, 0.715, 4.75, 1.6, 2, 0, -0.122, 1], # bus 6
    [5.06, 1.25, 0.232, 1.22, 0.715, 4.75, 1.6, 2, 0, 0, 1] # bus 7
]
parameters_= ["H", "X_d", "X_d_dash", "X_q", "X_q_dash", "T_d_dash", "T_q_dash", "D", "Rs", "P_m", "E_fd"]
matrix=pd.DataFrame(parameters, columns=parameters_)

#matrix.to_csv("src/parameters.csv", index=False)

# save yaml files for each of the parameters set
parameters=[dict(zip(parameters_, i)) for i in parameters]
#for i in range(len(parameters)):
#    with open(f"src/conf/params/machine{i+1}.yaml", "w") as file:
#        yaml.dump(parameters[i], file)

############################################################################################################
"""AVR parameters"""

avr_matrix = pd.DataFrame([[20, 0.2, 1.0, 0.314, 0.063, 0.35, 1.095]], columns=["KA", "TA", "KE", "TE", "KF", "TF", "V_ref"])
#avr_matrix.to_csv("src/avr_parameters.csv", index=False)

#with open(f"src/conf/params/avr.yaml", "w") as file:
#    yaml.dump({"K_A":20, "T_A":0.2, "K_E":1.0, "T_E":0.314, "K_F":0.063, "T_F":0.35, "V_ref":1.095}, file)
############################################################################################################
"""infinite bus parameters from phd thesis appendix"""

#[P_m,H,D,E_q_dash0,V,X_d_dash]=[0.1,2,2.5,1,1,2.5]

#infinite_bus_parameters={"P_m":P_m,"H":H,"D":D,"E_q_dash0":E_q_dash0,"V":V,"X_d_dash":X_d_dash}
#with open(f"src/conf/params/infinite_bus.yaml", "w") as file:
#    yaml.dump(infinite_bus_parameters, file)
############################################################################################################
"""system parameters"""

omega_s0=0
omega_B0=2*np.pi*50
theta_vs0=0
V0=1

system_parameters={"omega_s0":omega_s0,"omega_B0":omega_B0,"theta_vs0":theta_vs0,"V0":V0}
with open(f"src/conf/params/system.yaml", "w") as file:
    yaml.dump(system_parameters, file)
    
"""
$H$: Inertia constant (in seconds)

$X_d$: Direct axis synchronous reactance (in per unit)

$X_{d_dash}$: Direct axis transient reactance (in per unit)

$X_q$: Quadrature axis synchronous reactance (in per unit)

$X_{q_dash}$: Quadrature axis transient reactance (in per unit)

$T_{d_dash}$: Direct axis transient time constant (in seconds)

$T_{q_dash}$: Quadrature axis transient time constant (in seconds)

$D$: Damping coefficient (in per unit)

$R_s$: Stator resistance (in per unit)

$P_m$: Mechanical power input (in per unit)

$E_{fd}$: Excitation voltage (in per unit)
"""
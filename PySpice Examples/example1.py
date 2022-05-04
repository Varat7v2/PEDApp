####### CURRENT CONTROLLED VOLTAGE SOURCES ###################### 
import  PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Circuit with CCVS')
circuit.R(1, 1, 2,            2@u_Ω)
circuit.R(2, 2, circuit.gnd, 20@u_Ω)
circuit.R(3, 2, 3,            5@u_Ω)
circuit.R(4, 4, circuit.gnd, 10@u_Ω)
circuit.R(5, 4, 5,            2@u_Ω)
circuit.V(1, 1, circuit.gnd, 20@u_V)
circuit.V('test', 3, 4, 0@u_V)
# circuit.H(1, 5, circuit.gnd, 'Vtest', 8)
circuit.CCVS(1, 5, circuit.gnd, 'Vtest', 8)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()
for node in analysis.nodes.values(): 
	print('Node {}: {:4.1f} V'.format(str(node), float(node)))
for node in analysis.branches.values():
	print('Node {}: {:5.2f} A'.format(str(node), float(node))) 


##################################################################
# import  PySpice.Logging.Logging as Logging
# logger = Logging.setup_logging()
# from PySpice.Spice.Netlist import Circuit
# from PySpice.Unit import *


# circuit = Circuit('Circuit with CCCS')
# circuit.R(1, 2, 3, 6@u_Ω)
# circuit.R(2, 3, circuit.gnd, 8@u_Ω)
# circuit.R(3, 3, 4,           2@u_Ω)
# circuit.R(4, 4, circuit.gnd, 4@u_Ω)
# circuit.V(1, 1, circuit.gnd, 50@u_V)
# circuit.V('test', 1, 2,      0@u_V)
# circuit.I(1, circuit.gnd, 4, 5@u_A)
# circuit.F(1, 4, 3, 'Vtest', 3)
# circuit.CCCS(1, 4, 3, 'Vtest', 3)

# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# analysis = simulator.operating_point()

# for node in analysis.nodes.values(): 

# print('Node {}: {:4.1f} V'.format(str(node), float(node))) 

# for node in analysis.branches.values(): 

# print('Node {}: {:5.2f} A'.format(str(node), float(node))) 


# #######################################################


# import  PySpice.Logging.Logging as Logging
# logger = Logging.setup_logging()
# from PySpice.Spice.Netlist import Circuit
# from PySpice.Unit import *

# circuit = Circuit('Circuit with VCCS')
# circuit.R(1, 1, 2,           0.5@u_kΩ)
# circuit.R(2, 2, circuit.gnd, 1@u_kΩ)
# circuit.R(3, 2, 3,            2@u_kΩ)
# circuit.R(4, 3, circuit.gnd, 200@u_Ω)
# circuit.V(1, 1, circuit.gnd, 50@u_V)
# circuit.VCCS(1, circuit.gnd, 3, 2, circuit.gnd, 0.00133333333)

# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# analysis = simulator.operating_point()

# for node in analysis.nodes.values(): 

# print('Node {}: {:4.1f} V'.format(str(node), float(node))) 

# for node in analysis.branches.values(): 

# print('Node {}: {:5.2f} A'.format(str(node), float(node))) 


# #####################################################


# import PySpice.Logging.Logging as Logging
# logger = Logging.setup_logging()
# from PySpice.Spice.Netlist import Circuit
# from PySpice.Unit import *

# circuit = Circuit('Circuit with VCVS')
# circuit.R(1, 3, circuit.gnd, 5@u_Ω)
# circuit.R(2, 1, circuit.gnd, 6@u_Ω)
# circuit.R(3, 2, 3,           4@u_Ω)
# circuit.R(4, 3, 4,           8@u_Ω)
# circuit.R(5, 5, circuit.gnd, 15@u_Ω)
# circuit.V(1, 2, 1,           65@u_V)
# circuit.VCVS(1, 4, 5, 2, 3, 3)

# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# analysis = simulator.operating_point()

# for node in analysis.nodes.values(): 

# print('Node {}: {:4.1f} V'.format(str(node), float(node))) 

# for node in analysis.branches.values(): 

# print('Node {}: {:5.2f} A'.format(str(node), float(node))) 

#######################################################
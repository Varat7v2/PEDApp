### Resistor Bridge
import sys, os
import PySpice.Logging.Logging as Logging
import numpy as np
import math

logger = Logging.setup_logging()

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

from pathlib import Path
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Doc.ExampleTools import find_libraries

import matplotlib.pyplot as plt

# ### CHANGE SIMULATOR BASED ON OS SYSTEM
# if sys.platform == 'linux' or sys.platform == 'linux2':
#     PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'
# elif sys.platform == 'win32':
#     pass

# def format_output(analysis):
#     result_dict = dict()
#     # Loop through nodes
#     for node in analysis.nodes.values():
#         data_label = "%s" % str(node)
#         result_dict[data_label] = np.array(node)
#     return result_dict

# ### EXAMPLE 1
# circuit = Circuit('Resistor Bridge')
#
# circuit.V('input', 1, circuit.gnd, 10@u_V)
# circuit.R(1, 1, 2, 2@u_kΩ)
# circuit.R(2, 1, 3, 1@u_kΩ)
# circuit.R(3, 2, circuit.gnd, 1@u_kΩ)
# circuit.R(4, 3, circuit.gnd, 2@u_kΩ)
# circuit.R(5, 3, 2, 2@u_kΩ)
#
# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# analysis = simulator.operating_point()
#
# for node in analysis.nodes.values():
#     print('Node {}: {:4.1f} V'.format(str(node), float(node)))



# ### EXAMPLE 2 (VOLTAGE DIVIDER)
# ## CREATE THE CIRCUIT
# value = 5
# circuit = Circuit('Voltage Divider')

# ### ADD COMPONENTS TO THE CIRCUIT
# circuit.V(1, 1, circuit.gnd, value@u_V)
# circuit.R(1, 1, 2, 9@u_kOhm)
# circuit.R(2, 2, circuit.gnd, 1@u_kOhm)

# ### DISPLAY THE CIRUCIT ENTLIST
# print('The circuit netlist: \n\n', circuit)

# ### CREATE A SIMULATOR OBJECT
# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# print(simulator)

# ### RUN ANALYSIS
# analysis = simulator.operating_point()
# analysis_results = format_output(analysis)

# print(analysis_results)

# for node in (analysis['in'], analysis.out):
#     print('Node {} : {}V'.format(str(node), float(node)))

# # ## XYCE sensitivity analysis
# # analysis = simulator.dc_sensitivity('v(out)')
# # for element in analysis.elements.values():
# #     print(element, float(element))


# ### CREATE SUB-CIRCUIT
# class mySubCkt(SubCircuit):
#     __nodes__ = ('t_in', 't_out')
#     def __init__(self, name, r=1@u_kOhm):
#         SubCircuit.__init__(self, name, *self.__nodes__)
#         self.R(1, 't_in', 't_out', r)
#         self.Diode(1, 't_in','t_out', model='myDiode')
#         return


# #********************************************************************************
# ### EXAMPLE 3 (WITH DIODE)
# circuit = Circuit('Circuit with Diode')

# ## define 1n4148Ph (signal diode)
# circuit.model('myDiode', 'D', 
#                IS=4.352@u_nA,
#                RS=0.6458@u_Ohm,
#                BV=110@u_V,
#                IBV=0.0001@u_V,
#                N=1.906)

# # add components
# circuit.V('input', 1, circuit.gnd, 10@u_V)
# circuit.R(1, 1, 2, 9@u_kOhm)
# circuit.Diode(1, 2, 3, model='myDiode')
# circuit.subcircuit(mySubCkt('sub1', r=1@u_kOhm))
# circuit.X(1, 'sub1', 3, circuit.gnd )

# # print the circuit
# print(circuit)

# ### DISPLAY THE CIRUCIT ENTLIST
# print('The circuit netlist: \n\n', circuit)

# ### CREATE A SIMULATOR OBJECT
# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# print(simulator)

# ### RUN ANALYSIS
# analysis = simulator.operating_point()
# analysis_results = format_output(analysis)

# print(analysis_results)
# #***********************************************************************************




#************************************************************************************
### EXAMPLE 4 DC SWEEPS (WITH DIODE)
circuit = Circuit('DC SWEEP')

## define 1n4148Ph (signal diode)
circuit.model('myDiode', 'D', 
               IS=4.352@u_nA,
               RS=0.6458@u_Ohm,
               BV=110@u_V,
               IBV=0.0001@u_V,
               N=1.906)


# add components
circuit.V('INPUT', 1, 0, 10@u_V)
# circuit.Diode(1, 1, 2, model='myDiode')
# circuit.R(1, 2, circuit.gnd, 1@u_kOhm)

circuit.R(1, 1, 2, 1@u_kOhm)
circuit.R(2, 2, 0, 1@u_kOhm)
circuit.L(1, 2, 3, 100@u_uH)
circuit.C(1, 3, 0, 100@u_uF)
circuit.R(3, 3, 4, 1@u_kOhm)
circuit.R(4, 4, 0, 1@u_kOhm)

### DISPLAY THE CIRUCIT ENTLIST
print('The circuit netlist: \n\n', circuit)

### CREATE A SIMULATOR OBJECT
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
print(simulator)

### RUN ANALYSIS
analysis = simulator.dc(Vinput=slice(0,5,0.1))
# analysis_results = format_output(analysis)
# print(analysiss_results)
print(analysis)

### PLOT GRAPH
fig1 = plt.figure()
plt.plot(np.array(analysis['1']), np.array(analysis['2']))
plt.xlabel('Input voltage (node 1)')
plt.ylabel('Output voltage (node 2')
plt.show()
# fig1.savefig('dc_sweep.png', dpi=300)
#*******************************************************************************



# #******************************************************************************
# ### EXAMPLE 5: INCLUDING MODELS AND LIBRARIES
# circuit = Circuit('Including models and libraries')

# # add components
# circuit.V('input', 1, circuit.gnd, 10@u_V)
# circuit.R(1, 2, circuit.gnd, 1@u_kOhm)

# # ### METHOD 1
# # # spice_library = SpiceLibrary('/home/varat/Documents/LTspiceXVII/lib/cmp/')

# # # libraries_path = find_libraries()
# # path = '/home/varat/Documents/LTspiceXVII/lib/cmp/standard.dio'
# # spice_library = SpiceLibrary(path)
# # circuit.include(spice_library['1N4148'])
# # circuit.X('importDiode', '1N4148', 1, 2)

# ### METHOD 2
# path = '/home/varat/Documents/LTspiceXVII/lib/cmp/standard.dio'
# circuit.include(path)
# circuit.X('importDiode', '1N4148', 1, 2)

# # ### METHOD 3
# # new_line = '.include lib/1N4148.lib'
# # circuit.raw_spice += new_line + os.linesep
# # circuit.X('importDiode', '1N4148', 1, 2)

# # print the circuit
# print(circuit)

# ### DISPLAY THE CIRUCIT ENTLIST
# print('The circuit netlist: \n\n', circuit)

# ### CREATE A SIMULATOR OBJECT
# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# print(simulator)

# # ### RUN ANALYSIS
# analysis = simulator.dc(Vinput=slice(0,5,0.1))
# # analysis_results = format_output(analysis)
# # print(analysiss_results)
# print(analysis)

# ### PLOT GRAPH
# fig1 = plt.figure()

# plt.plot(np.array(analysis['1']), np.array(analysis['2']))
# plt.xlabel('Input voltage (node 1)')
# plt.ylabel('Output voltage (node 2')

# plt.show()
# # fig1.savefig('dc_sweep.png', dpi=300)
# #********************************************************************************

# ### EXAMPLE 5: TRANSIENT AND AC ANALYSIS
# circuit = Circuit('AC Transient Analysis')
# circuit.model('myDiode', 'D', 
#                IS=4.352@u_nA,
#                RS=0.6458@u_Ohm,
#                BV=110@u_V,
#                IBV=0.0001@u_V,
#                N=1.906)

# # add components
# Vac = circuit.SinusoidalVoltageSource('input', 'n1', circuit.gnd, amplitude=1@u_V, frequency=100@u_Hz)
# # circuit.V('input', 'n1', circuit.gnd, 10@u_V)
# R = circuit.R(1, 'n1', 'n2', 1@u_kOhm)
# C = circuit.C(1, 'n2', circuit.gnd, 1@u_uF)

# circuit.Diode(1, 'n2', 'n3', model='myDiode')
# # path = 'lib/1N4148.lib'
# # circuit.include(path)
# # circuit.X('importDiode', '1N4148', 'n2', 'n3')

# circuit.R(2, 'n3', circuit.gnd, 1@u_kOhm)

# # print the circuit
# print(circuit)

# ### DISPLAY THE CIRUCIT ENTLIST
# print('The circuit netlist: \n\n', circuit)

# ### CREATE A SIMULATOR OBJECT
# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# print(simulator)

# # # ### RUN DC ANALYSIS
# # analysis = simulator.dc(Vinput=slice(-3,3,0.1))
# # print(analysis)

# # ### TRANSISENT ANALYSIS
# # analysis = simulator.transient(step_time=Vac.period/10, end_time=Vac.period)

# ### AC ANALYSIS
# analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_MHz, variation='dec', number_of_points=10)

# break_frequency = 1 / (2*math.pi * float(R.resistance * C.capacitance))
# print('Break frequency = {:.1f} Hz'.format(break_frequency))

# ## BODE PLOT
# fig, axes = plt.subplots(2, figsize=(10,20))
# plt.title('Bode Diagram of a Low-Pass RC Filter')
# bode_diagram(axes=axes,
#              frequency=analysis.frequency,
#              gain=20*np.log10(np.absolute(analysis.n2)),
#              phase=np.angle(analysis.n2, deg=False),
#              marker='.',
#              color='blue',
#              linestyle='-'
#              )

# for ax in axes:
#     ax.axvline(x=break_frequency, color='red')
# plt.tight_layout()

# # ### PLOT GRAPH
# # fig = plt.figure()
# # plt.plot(np.array(analysis.time), np.array(analysis['n1']), label='V(n1)')
# # plt.plot(np.array(analysis.time), np.array(analysis['n2']), label='V(n2)')
# # plt.plot(np.array(analysis.time), np.array(analysis['n3']), label='V(n3)')
# # plt.legend()
# # plt.xlabel('Time')
# # plt.ylabel('Voltage')

# plt.show()
# fig.savefig('bode_plot.png', dpi=300)
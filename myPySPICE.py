import sys, os
import PySpice.Logging.Logging as Logging
import numpy as np
import math
import random

logger = Logging.setup_logging()

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Plot.BodeDiagram import bode_diagram

import matplotlib.pyplot as plt

# ### CHANGE SIMULATOR BASED ON OS SYSTEM
# if sys.platform == 'linux' or sys.platform == 'linux2':
#     PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'
# elif sys.platform == 'win32':
#     pass

class myPYSPICE(object):
    def __init__(self, title):
        self.title = title

    def format_output(self, analysis):
        result_dict = dict()
        # Loop through nodes
        for node in analysis.nodes.values():
            data_label = "%s" % str(node)
            result_dict[data_label] = np.array(node)
        return result_dict

    def simulate_ckt(self, netlist_dict):
        f_netlist = open('./data/UH_ECD.cir', 'w')
        f_netlist.write('*Generated netlist for the circuit diagram.\n')
        ## CREATE THE CIRCUIT
        circuit = Circuit(self.title)
        ### ADD COMPONENTS TO THE CIRCUIT
        ### INITIALIZE THE COMPONENTS COUNT
        R_count = 0
        L_count = 0
        C_count = 0
        D_count = 0
        S_count = 0
        G_count = 0
        SW_count = 0
        MOSFET_count = 0
        TRANS_count = 0

        for k,v in netlist_dict.items():
            branch = k
            component = v[0][0]
            terminal1 = v[1][0]
            terminal2 = v[1][1]

            ### ASSIGNING NODE NUMBERED GREATER THAN '5' TO '0'
            if terminal1 >= 5:
                terminal1 = 0
            if terminal2 >= 5:
                terminal2 = 0

            ### REPLACING '0' TERMINAL TO CIRCUIT.GND
            if terminal1 == 0:
                terminal1 = circuit.gnd
            if terminal2 == 0:
                terminal2 = circuit.gnd

            ### RANDOM VALUES ASSIGNED TO THE COMPONENTS (LATER REPLACE BY TEXT RECOGNITION ALGO)
            S_value = random.randint(5,12)      # value in Volts(V)
            R_value = random.randint(1,100)     # value in Kilo ohms (kOhm)
            L_value = random.randint(1,100)     # value in micro henry (uH)
            C_value = random.randint(1,100)     # value in micro farads (uF)

            if component == 'source':
                S_count += 1
                circuit.V(S_count, terminal1, terminal2, S_value@u_V)
                f_netlist.write('V{} {} {} {}\n'.format(S_count, terminal1, terminal2, S_value))
            elif component == 'resistor':
                R_count += 1
                circuit.R(R_count, terminal1, terminal2, R_value@u_kOhm)
                f_netlist.write('R{} {} {} {}\n'.format(R_count, terminal1, terminal2, R_value))
            elif component == 'inductor':
                L_count += 1
                circuit.L(L_count, terminal1, terminal2, L_value@u_uH)
                f_netlist.write('L{} {} {} {}\n'.format(L_count, terminal1, terminal2, L_value))
            elif component == 'capacitor':
                C_count += 1
                circuit.C(C_count, terminal1, terminal2, C_value@u_uF)
                f_netlist.write('C{} {} {} {}\n'.format(C_count, terminal1, terminal2, C_value))
            elif component == 'diode':
                pass
            elif component == 'mosfet':
                pass
            elif component == 'switch':
                pass
            elif component == 'transistor':
                pass
        f_netlist.write('.op')
        f_netlist.close()
        print('INFO: Netlist for the circuit diagram written!')
        # print(circuit)
        ### CREATE A SIMULATOR OBJECT
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
        ### RUN ANALYSIS
        analysis = simulator.operating_point()
        analysis_results = self.format_output(analysis)
        # print(analysis_results)
        return circuit, analysis_results

if __name__ == '__main__':
    PROJECT = 'Voltage Divider'
    ckt = myPYSPICE(PROJECT)
    ### Note: [WARNING] Terminal order in netlist is from anode(+) to cathode(-)
    netlist_dict = {
                       '0': [['source'],   [1, 0]], 
                       '1': [['resistor'], [1, 2]], 
                       '2': [['resistor'], [2, 0]],
                    }
    circuit, analysis_results = ckt.simulate_ckt(netlist_dict)
    print('Circuit:\n', circuit)
    print('Simulation Results:\n', analysis_results)
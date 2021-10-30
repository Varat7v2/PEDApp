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
        ### NETLIST FILE FOR PYSPICE SIMULATION
        PySpice_netlist = open('./data/UH_ECD.cir', 'w')
        PySpice_netlist.write('*Generated netlist for the circuit diagram.\n')

        ### NETLIST FILE FOR KICAD-PCBNEW-BASED PCD DESIGNING
        PCBNEW_netlist = open('./data/UH_ECD_PCB.net', 'w')
        PCBNEW_netlist.write('(export (version D)\n')
        PCBNEW_netlist.write('### FIXED SECTION\n \
                               ### DESIGN BLOCK\n \
                               ### LIBRARIES BLOCK\n')
        PCBNEW_netlist.write('(libraries\n \
                               (library (logical Device)\n \
                               (uri /usr/share/kicad/library/Device.lib))\n \
                               (library (logical pspice)\n \
                               (uri /usr/share/kicad/library/pspice.lib)))\n')
        PCBNEW_netlist.write('### VARIABLE SECTION\n \
                               ### COMPONENT BLOCK\n')
        PCBNEW_netlist.write('(components\n')

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

        COMPONENT_COUNT = 0
        LABEL = list()
        REF = list()
        VALUE = list()
        NODE_COORDS = list()
        FOOTPRINT = list()
        LIBSOURCE_LIB = list()
        LIBSOURCE_PART = list()
        LIBSOURCE_DESP = list()
        TSTAMP = list()

        for k,v in netlist_dict.items():
            COMPONENT_COUNT += 1
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

            LABEL.append(component)
            NODE_COORDS.append(tuple((terminal1, terminal2)))
            TSTAMP.append('{}'.format(COMPONENT_COUNT))

            if component == 'source':
                S_count += 1
                circuit.V(S_count, terminal1, terminal2, S_value@u_V)
                PySpice_netlist.write('V{} {} {} {}\n'.format(S_count, terminal1, terminal2, S_value))
                VALUE.append(S_value)
                REF.append('V{}'.format(S_count))
                FOOTPRINT.append('Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical')
                LIBSOURCE_LIB.append('Simulation_SPICE')
                LIBSOURCE_PART.append('VDC')
                LIBSOURCE_DESP.append('Voltage source, DC')
                PCBNEW_netlist.write('### COMPONENT_{}\n \
                                      (comp\n \
                                      (ref V{})\n \
                                      (value {})\n \
                                      (footprint Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical)\n \
                                      (datasheet ~)\n \
                                      (libsource (lib Simulation_SPICE) (part VDC) (description "Voltage source, DC"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                S_count,
                                                                S_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'resistor':
                R_count += 1
                circuit.R(R_count, terminal1, terminal2, R_value@u_kOhm)
                PySpice_netlist.write('R{} {} {} {}\n'.format(R_count, terminal1, terminal2, R_value))
                VALUE.append(R_value)
                REF.append('R{}'.format(R_count))
                FOOTPRINT.append('Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal')
                LIBSOURCE_LIB.append('Device')
                LIBSOURCE_PART.append('R')
                LIBSOURCE_DESP.append('Resistor')
                PCBNEW_netlist.write('### COMPONENT_{}\n \
                                      (comp\n \
                                      (ref R{})\n \
                                      (value {})\n \
                                      (footprint Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P15.24mm_Horizontal)\n \
                                      (datasheet ~)\n \
                                      (libsource (lib Device) (part R) (description Resistor))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                R_count,
                                                                R_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'inductor':
                L_count += 1
                circuit.L(L_count, terminal1, terminal2, L_value@u_uH)
                PySpice_netlist.write('L{} {} {} {}\n'.format(L_count, terminal1, terminal2, L_value))
                VALUE.append(L_value)
                REF.append('L{}'.format(L_count))
                FOOTPRINT.append('Inductor_THT:L_Axial_L5.3mm_D2.2mm_P10.16mm_Horizontal_Vishay_IM-1')
                LIBSOURCE_LIB.append('pspice')
                LIBSOURCE_PART.append('INDUCTOR')
                LIBSOURCE_DESP.append('Inductor symbol for simulation only')
                PCBNEW_netlist.write('### COMPONENT_{}\n \
                                      (comp\n \
                                      (ref L{})\n \
                                      (value {})\n \
                                      (footprint Inductor_THT:L_Axial_L5.3mm_D2.2mm_P10.16mm_Horizontal_Vishay_IM-1)\n \
                                      (datasheet ~)\n \
                                      (libsource (lib pspice) (part INDUCTOR) (description "Inductor symbol for simulation only"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                L_count,
                                                                L_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'capacitor':
                C_count += 1
                circuit.C(C_count, terminal1, terminal2, C_value@u_uF)
                PySpice_netlist.write('C{} {} {} {}\n'.format(C_count, terminal1, terminal2, C_value))
                VALUE.append(C_value)
                REF.append('C{}'.format(C_count))
                FOOTPRINT.append('Capacitor_THT:CP_Radial_D5.0mm_P2.50mm')
                LIBSOURCE_LIB.append('Device')
                LIBSOURCE_PART.append('C')
                LIBSOURCE_DESP.append('Unpolarized capacitor')
                PCBNEW_netlist.write('### COMPONENT_{}\n \
                                      (comp\n \
                                      (ref C{})\n \
                                      (value {})\n \
                                      (footprint Capacitor_THT:CP_Radial_D5.0mm_P2.50mm)\n \
                                      (datasheet ~)\n \
                                      (libsource (lib Device) (part C) (description "Unpolarized capacitor"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                C_count,
                                                                C_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'diode':
                pass
            elif component == 'mosfet':
                pass
            elif component == 'switch':
                pass
            elif component == 'transistor':
                pass
        PySpice_netlist.write('.op')
        PySpice_netlist.close()
        print('INFO: Netlist for the circuit diagram written!!!')

        ### WRAPING UP PCBNEW NETLIST FILE
        PCBNEW_netlist.write(')\n)\n')
        PCBNEW_netlist.close()
        print('INFO: PCBNEW Netlist for PCB design written!!!')

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
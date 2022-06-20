import sys, os
import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries
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

import config as myconfig

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
                               (uri /usr/share/kicad/library/Device.libraries))\n \
                               (library (logical pspice)\n \
                               (uri /usr/share/kicad/library/pspice.libraries)))\n')
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
        MOS_count = 0
        IGBT_count = 0
        BJT_count = 0

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
        Diode_Type = '1N4148'

        ### COMPONENTS LIBRARIES PATH
        libraries_path = find_libraries()
        # libraries_path = 'libraries'
        spice_library = SpiceLibrary(libraries_path)

        ### Transient conditions
        steptime=1@u_us
        switchingtime=50@u_ms
        finaltime=250@u_ms

        ### MOSFET conditions
        MOSFET_TYPE = 'irf150'
        duty_cycle = float(random.randint(40, 60)) / 100     # ratio = Vout/Vin (As for now assumped; Need to modify later)
        fs = random.randint(20, 50)@u_kHz                    # Switching frequency
        Ts = fs.period
        Ton = duty_cycle * Ts                                # T_ON

        print('Dutycycle = ', duty_cycle)
        print('Switching frequency = ', fs)

        ### DEFINE DIODE MODEL (Manual Method)
        # DIODE_MODEL = '.model 1N4148 D (BV=110V IBV=0.0001V IS=4.352nA N=1.906 RS=0.6458Ohm)'
        # circuit.model( Diode_Type, 'D',
        #                IS=4.352@u_nA,
        #                RS=0.6458@u_Ohm,
        #                BV=110@u_V,
        #                IBV=0.0001@u_V,
        #                N=1.906)

        print('\n\n'+ '*'*200)
        print(netlist_dict)
        print('*'*200+'\n\n')

        print('\n\n'+ '*'*200)
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

            ### DISPLAY COMPONENTS AND THEIR TERMINALS
            # print(component + ': ' + str(terminal1) + ' ' + str(terminal2))

            ### RANDOM VALUES ASSIGNED TO THE COMPONENTS (LATER REPLACE BY TEXT RECOGNITION ALGO)
            S_value = random.randint(15,24)      # value in Volts(V)
            R_value = random.uniform(0.5, 2.5)     # value in Kilo ohms (kOhm)
            L_value = random.randint(100, 300)     # value in micro henry (uH)
            C_value = random.randint(400, 800)     # value in micro farads (uF)

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
                                      (libsource (libraries Simulation_SPICE) (part VDC) (description "Voltage source, DC"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                S_count,
                                                                S_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'resistor':
                R_count += 1
                circuit.R(R_count, terminal1, 'xr{}'.format(R_count), R_value@u_Ohm)
                circuit.V('branch_R', 'xr{}'.format(R_count), terminal2, 0@u_V)
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
                                      (libsource (libraries Device) (part R) (description Resistor))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                R_count,
                                                                R_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'inductor':
                L_count += 1
                circuit.L(L_count, terminal1, 'xl{}'.format(L_count), L_value@u_uH)
                # For current measurement purpose
                circuit.V('branch_L', 'xl{}'.format(L_count), terminal2, 0@u_V)
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
                                      (libsource (libraries pspice) (part INDUCTOR) (description "Inductor symbol for simulation only"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                L_count,
                                                                L_value,
                                                                COMPONENT_COUNT,
                                                            )
                                    )
            elif component == 'capacitor':
                # if C_count < 1:
                #     IC=0V
                C_count += 1
                circuit.C(C_count, terminal1, 'xc{}'.format(C_count), C_value@u_uF)
                circuit.V('branch_C', 'xc{}'.format(C_count), terminal2, 0@u_V)
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
                                      (libsource (libraries Device) (part C) (description "Unpolarized capacitor"))\n \
                                      (sheetpath (names /) (tstamps /))\n \
                                      (tstamp {}))\n'.format(   COMPONENT_COUNT,
                                                                C_count,
                                                                C_value,
                                                                COMPONENT_COUNT,
                                                             )
                                    )
            elif component == 'diode':
                if D_count < 1:
                    DIODE_TYPE = '1N4002'
                    circuit.include(spice_library[DIODE_TYPE])
                D_count += 1
                circuit.X('D{}'.format(D_count), DIODE_TYPE, terminal1, 'xd{}'.format(D_count))
                circuit.V('branch_D', 'xd{}'.format(D_count), terminal2, 0@u_V)
                # circuit.Diode(D_count, terminal1, terminal2, model=Diode_Type)
                PySpice_netlist.write('D{} {} {} 1N4148\n'.format(D_count, terminal1, terminal2))
            
            elif component == 'mosfet' or component == 'switch':
                if MOS_count < 1:
                    circuit.include(spice_library[MOSFET_TYPE])
                MOS_count += 1
                R_count += 1
                circuit.X('Q{}'.format(MOS_count), MOSFET_TYPE, terminal1, 'Gate', 'xmos{}'.format(MOS_count))
                circuit.V('branch_mos', 'xmos{}'.format(MOS_count), terminal2, 0@u_V)
                circuit.R(R_count, 'Gate', 'Clock', 1@u_Ω)
                circuit.PulseVoltageSource('Pulse',             # Name
                                           'Clock',             # Node +ve
                                           circuit.gnd,         # Node -ve
                                           initial_value=0,
                                           pulsed_value=S_value,
                                           pulse_width=Ton,
                                           period=Ts)
            elif False:
            # elif component == 'switch':
                # if SW_count < 1:
                #     ### Modelling SWITCH
                S_count += 1
                R_count += 1
                SW_count += 1
                circuit.model('switch', 'SW', Ron=1@u_mΩ, Roff=1@u_GΩ)
                circuit.PulseVoltageSource(S_count, 'posA', circuit.gnd, initial_value=0, pulsed_value=1,
                                           pulse_width=finaltime, period=finaltime, delay_time=switchingtime)
                circuit.R('testA', 'posA', circuit.gnd, 1@u_kΩ)
                #### VoltageControlledSwitch(name, n+, n-, nc+, nc-, model)
                circuit.VoltageControlledSwitch(SW_count, terminal1, terminal2, 'posA', circuit.gnd, model='switch')
                PySpice_netlist.write('R{} posA 0 1k\n'.format(R_count))
                PySpice_netlist.write('S{} {} {} posA 0 SW\n'.format(SW_count, terminal1, terminal2))
                PySpice_netlist.write('V{} {} {} PULSE(0 1 50m 1u 1u 250m 250m)\n'.format(S_count, 'posA', circuit.gnd))

            elif component == 'transistor':
                pass

        ### FOR THE TEST PURPOSE ONLY ### TODO: REMOVE LATER
        circuit.R('test', 3, 4, 0.5@u_Ohm)
        print('*'*200+'\n\n')
        
        PySpice_netlist.write('.libraries /home/varat/Documents/LTspiceXVII/libraries/cmp/standard.dio\n')
        PySpice_netlist.write('.options TEMP = 25C\n')
        PySpice_netlist.write('.options TNOM = 25C\n')
        PySpice_netlist.write('.tran 100m\n')
        PySpice_netlist.write('.model switch SW(Ron=1m Roff=100Meg Vt=0.5 Vh=0)\n')
        PySpice_netlist.write('.end')
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
        if myconfig.ANALYSIS == 'OPERATING_POINT':
            analysis = simulator.operating_point()
            print('SIMULATION RESULT')
            for node in analysis.nodes.values():
                print('Node {}: {:4.2f} V'.format(str(node), float(node)))
            for node in analysis.branches.values():
                print('Node {}: {:5.5f} A'.format(str(node), float(node)))

        if myconfig.ANALYSIS == 'TRANSIENT':
            IC=0@u_V
            # simulator.initial_condition(output=IC)
            analysis = simulator.transient(step_time=Ts/150, end_time=Ts*300)
            
        return circuit, analysis

if __name__ == '__main__':
    PROJECT = 'Voltage Divider'
    ckt = myPYSPICE(PROJECT)
    ### Note: [WARNING] Terminal order in netlist is from anode(+) to cathode(-)
    netlist_dict = {
                       '0': [['source'],   [1, 0]], 
                       '1': [['resistor'], [1, 2]], 
                       '2': [['resistor'], [2, 0]],
                    }
    circuit, analysis = ckt.simulate_ckt(netlist_dict)
    
    for node in analysis.nodes.values():
        print('Node {}: {:4.1f} V'.format(str(node), float(node)))

    for node in analysis.branches.values():
        print('Node {}: {5.2f} A'.format(str(node), float(node)))

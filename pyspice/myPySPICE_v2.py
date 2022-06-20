import sys, os
import PySpice.Logging.Logging as Logging
import numpy as np
import math
import random
import config as myconfig
import matplotlib.pyplot as plt

logger = Logging.setup_logging()

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Doc.ExampleTools import find_libraries

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

    def simulate_ckt(self, netlist_dict, converter):
        ## NETLIST FILE FOR PYSPICE SIMULATION
        # if not os.path.exists(self.filename):
        PySpice_netlist = open('{}/{}'.format(myconfig.DATA, self.title), 'w')
        PySpice_netlist.write('*Generated netlist for the circuit diagram.\n')

        ### NETLIST FILE FOR KICAD-PCBNEW-BASED PCD DESIGNING
        PCBNEW_netlist = open('{}/{}_PCB.net'.format(myconfig.DATA, self.title), 'w')
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
        # Initialize the component count
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
        ESR_count = 0

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

        ### COMPONENTS LIBRARIES PATH
        # lib_path = find_libraries()
        spice_library = SpiceLibrary(myconfig.LIB_PATH)

        ### Transient conditions (Simulation runtime)
        steptime=1e-3
        endtime =0.5

        ### Switching parameters
        # fs and Ts based on the IEEE paper by Krishnamoorthy et. al. (2019)
        duty_cycle = random.uniform(0.4, 0.6)    # ratio = Vout/Vin (As for now assumped; Need to modify later)
        fs = random.randint(20, 50)@u_kHz        # Switching frequency
        Ts = fs.period
        Ton = duty_cycle * Ts                    # T_ON

        print('Dutycycle = ', duty_cycle)
        print('Switching frequency = ', fs)

        # print('\n\n'+ '*'*200)
        # print(netlist_dict)
        # print('*'*200+'\n\n')
        # print('\n\n'+ '*'*200)
        
        print('[INFO]: Updated netlist for Spice Analysis')
        for k,v in netlist_dict.items():
            COMPONENT_COUNT += 1
            component = v[0][0]
            terminal1 = v[1][0]
            terminal2 = v[1][1]

            ### TODO: Later change this to act automatically as per component direction
            # Rearranging nodes terminal as per the convert types
            if (converter=='Buck Converter' and component=='diode') \
                or (converter=='Buck-Boost Converter' and (component=='diode' or component=='capacitor')):
                    temp = terminal1
                    terminal1 = terminal2
                    terminal2 = temp

            # Assigning node number greater than 5 to circuit.gnd
            if terminal1 >= 5:
                terminal1 = circuit.gnd
            if terminal2 >= 5:
                terminal2 = circuit.gnd

            if myconfig.DEBUG_PEDAPP:
                ## DISPLAY COMPONENTS AND THEIR TERMINALS - Easy to understand
                print('\t\t {}: {} {}'.format(component, str(terminal1), str(terminal2)))

            ### RANDOM VALUES ASSIGNED TO THE COMPONENTS (LATER REPLACE BY TEXT RECOGNITION ALGO)
            # Based on the IEEE paper by Krishnamoorthy et. al. (2019)
            S_value = random.randint(45,  54)               # value in Volts(V)
            R_value = round(random.uniform(0.5, 2.5), 2)    # value in ohms (Ohm)
            L_value = random.randint(100, 300)              # value in micro henry (uH)
            C_value = random.randint(400, 800)              # value in micro farads (uF)

            LABEL.append(component)
            NODE_COORDS.append(tuple((terminal1, terminal2)))
            TSTAMP.append('{}'.format(COMPONENT_COUNT))

            if component == 'source':
                S_count += 1
                circuit.V(S_count, terminal1, terminal2, S_value@u_V)

                ### PySpice Netlist
                PySpice_netlist.write('V{} {} {} {}\n'.format(S_count, terminal1, terminal2, S_value))

                ### PCBNEW Netlist
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

                ### PySpice Netlist
                PySpice_netlist.write('R{} {} {} {}\n'.format(R_count, terminal1, terminal2, R_value))

                ### PCBNEW List
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
                if myconfig.ESR_FLAG:
                    ESR_count += 1
                    circuit.R('ESR{}'.format(ESR_count), 'xl{}'.format(L_count), 'xl_esr', myconfig.ESR_Value@u_Ohm)
                    circuit.V('branch_L', 'xl_esr', terminal2, 0@u_V)
                else:
                    circuit.V('branch_L', 'xl{}'.format(L_count), terminal2, 0@u_V)

                ### PySpice Netlist
                PySpice_netlist.write('L{} {} {} {}\n'.format(L_count, terminal1, terminal2, L_value))

                ### PCBNEW Netlist
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
                if myconfig.ESR_FLAG:
                    ESR_count += 1
                    circuit.R('ESR{}'.format(ESR_count), 'xc{}'.format(L_count), 'xc_esr', myconfig.ESR_Value@u_Ohm)
                    circuit.V('branch_C', 'xc_esr', terminal2, 0@u_V)
                else:
                    circuit.V('branch_C', 'xc{}'.format(C_count), terminal2, 0@u_V)

                ### PySpice Netlist
                PySpice_netlist.write('C{} {} {} {}\n'.format(C_count, terminal1, terminal2, C_value))

                ### PCBNEW Netlist
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
                    circuit.include(spice_library[myconfig.DIODE_TYPE])
                D_count += 1
                circuit.X('D{}'.format(D_count), myconfig.DIODE_TYPE, terminal1, 'xd{}'.format(D_count))
                circuit.V('branch_D', 'xd{}'.format(D_count), terminal2, 0@u_V)

                ### PySpice Netlist
                PySpice_netlist.write('D{} {} {} {}\n'.format(D_count, terminal1, terminal2, myconfig.DIODE_TYPE))
            
            elif component == 'mosfet' or component == 'switch':
                if MOS_count < 1:
                    circuit.include(spice_library[myconfig.MOSFET_TYPE])
                MOS_count += 1
                R_count += 1
                # Exp: circuit.X('Q', 'irf150', 'in', 'gate', 'source')
                circuit.X('Q{}'.format(MOS_count), myconfig.MOSFET_TYPE, terminal1, 'Gate', 'xmos{}'.format(MOS_count))
                circuit.V('branch_mos', 'xmos{}'.format(MOS_count), terminal2, 0@u_V)
                if myconfig.ESR_FLAG:
                    ESR_count += 1
                    circuit.R('ESR{}'.format(ESR_count), 'Gate', 'Clock', myconfig.ESR_Value@u_Ω)
                    circuit.PulseVoltageSource('Pulse',             # Name
                                               'Clock',             # Node +ve
                                               terminal2,           # Node -ve
                                               initial_value=0,
                                               pulsed_value=2.*S_value,
                                               rise_time=10e-9,
                                               fall_time=10e-9,
                                               delay_time=0,
                                               pulse_width=Ton,
                                               period=Ts)
                else:
                    circuit.PulseVoltageSource('Pulse',             # Name
                                               'Gate',             # Node +ve
                                               terminal2,           # Node -ve
                                               initial_value=0,
                                               pulsed_value=2.*S_value,
                                               rise_time=10e-9,
                                               fall_time=10e-9,
                                               delay_time=0,
                                               pulse_width=Ton,
                                               period=Ts)
            ### FOR SWITCH ONLY
            # elif False:
            # # elif component == 'switch':
            #     # if SW_count < 1:
            #     #     ### Modelling SWITCH
            #     S_count += 1
            #     R_count += 1
            #     SW_count += 1
            #     circuit.model('switch', 'SW', Ron=1@u_mΩ, Roff=1@u_GΩ)
            #     circuit.PulseVoltageSource(S_count, 'posA', circuit.gnd, initial_value=0, pulsed_value=1,
            #                                pulse_width=finaltime, period=finaltime, delay_time=switchingtime)
            #     circuit.R('testA', 'posA', circuit.gnd, 1@u_kΩ)
            #     #### VoltageControlledSwitch(name, n+, n-, nc+, nc-, model)
            #     circuit.VoltageControlledSwitch(SW_count, terminal1, terminal2, 'posA', circuit.gnd, model='switch')
            #     # PySpice_netlist.write('R{} posA 0 1k\n'.format(R_count))
            #     # PySpice_netlist.write('S{} {} {} posA 0 SW\n'.format(SW_count, terminal1, terminal2))
            #     # PySpice_netlist.write('V{} {} {} PULSE(0 1 50m 1u 1u 250m 250m)\n'.format(S_count, 'posA', circuit.gnd))

            elif component == 'transistor':
                pass

        ### FOR THE TEST PURPOSE ONLY ### TODO: REMOVE LATER
        circuit.R('empty_branch', 3, 4, myconfig.ESR_Value@u_Ohm)
        print('*'*200+'\n\n')

        ### PySpice Netlist
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
            # IC=0@u_V
            # simulator.initial_condition(output=IC)
            analysis = simulator.transient(step_time=steptime, end_time=endtime)

        return circuit, analysis

### Test Code
if __name__ == '__main__':
    ckt = myPYSPICE(myconfig.PROJECT)
    ### Note: [WARNING] Terminal order in netlist is from anode(+) to cathode(-)
    netlist_dict = {
                        '0': [['switch'],       [0, 1]],
                        '1': [['inductor'],     [1, 2]],
                        '2': [['source'],       [0, 4]],
                        '3': [['capacitor'],    [2, 6]],
                        '4': [['diode'],        [1, 5]],
                        '5': [['resistor'],     [3, 7]],
                    }
    circuit, analysis = ckt.simulate_ckt(netlist_dict)

    print('[INFO]: Circuit analysis results: ', circuit)
    
    # for node in analysis.nodes.values():
    #     print('Node {}: {:4.1f} V'.format(str(node), float(node)))
    #
    # for node in analysis.branches.values():
    #     print('Node {}: {5.2f} A'.format(str(node), float(node)))

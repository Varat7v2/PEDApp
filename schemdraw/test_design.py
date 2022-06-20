# -*- coding: utf-8 -*-

from skidl import *


def _home_varat_myPhD_PEDApp_PROJECT_ECD_skidl_examples_py():

    #===============================================================================
    # Component templates.
    #===============================================================================

    NO_LIB_R_R_0201_0603Metric = Part('NO_LIB', 'R', dest=TEMPLATE, footprint='R_0201_0603Metric')


    #===============================================================================
    # Component instantiations.
    #===============================================================================

    R1 = NO_LIB_R_R_0201_0603Metric(ref='R1', value='1K')

    R2 = NO_LIB_R_R_0201_0603Metric(ref='R2', value='500')

    R3 = NO_LIB_R_R_0201_0603Metric(ref='R3', value='1K')

    R4 = NO_LIB_R_R_0201_0603Metric(ref='R4', value='500')

    R5 = NO_LIB_R_R_0201_0603Metric(ref='R5', value='1K')

    R6 = NO_LIB_R_R_0201_0603Metric(ref='R6', value='500')


    #===============================================================================
    # Net interconnections between instantiated components.
    #===============================================================================

    Net('GND').connect(R2['2'], R4['2'], R6['2'])

    Net('IN').connect(R1['1'])

    Net('N$1').connect(R1['2'], R2['1'], R3['1'])

    Net('N$2').connect(R3['2'], R4['1'], R5['1'])

    Net('OUT').connect(R5['2'], R6['1'])


#===============================================================================
# Instantiate the circuit and generate the netlist.
#===============================================================================

if __name__ == "__main__":
    _home_varat_myPhD_PEDApp_PROJECT_ECD_skidl_examples_py()
    generate_netlist()

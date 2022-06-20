import os

os.environ['KICAD_SYMBOL_DIR'] = '/usr/share/kicad/library'

from skidl import *

active_lib = 'simple'

test_resistor_0, = 1 * Part(active_lib,
					 		'resistor',
					 		TEMPLATE,
					 		footprint='Resistor_SMD:R_0201_0603Metric')

test_resistor_1, = 1 * Part(active_lib,
					 		'resistor',
					 		TEMPLATE,
					 		footprint='Resistor_SMD:R_0201_0603Metric')

test_resistor_0.value = '100'
test_resistor_1.value = '220'

gnd = Net('GND')
vin = Net('VIN')

gnd += test_resistor_0[1]
gnd += test_resistor_1[1]

vin += test_resistor_0[2], test_resistor_1[2]

generate_netlist(file_='test.net')
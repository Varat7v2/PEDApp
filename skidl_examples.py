# from skidl import *
import os, sys
os.environ['KICAD_SYMBOL_DIR'] = '/usr/share/kicad/library'

# active_lib = "simple"
# resistor_0, = 1 * Part(active_lib, 'resistor', TEMPLATE, footprint='R_0201_0603Metric')

# generate_netlist(file_='skidl_ex1.net')

### EXAMPLE 1
# from skidl import *

# gnd = Net('GND')  # Ground reference.
# vin = Net('VI')   # Input voltage to the divider.
# vout = Net('VO')  # Output voltage from the divider.
# r1, r2 = 2 * Part('/usr/share/kicad/library/Device.lib', 'R', TEMPLATE)  # Create two resistors.
# r1.value, r1.footprint = '1K',  'R_0201_0603Metric'  # Set resistor values
# r2.value, r2.footprint = '500', 'R_0201_0603Metric'  # and footprints.
# r1[1] += vin      # Connect the input to the first resistor.
# r2[2] += gnd      # Connect the second resistor to ground.
# vout += r1[2], r2[1]  # Output comes from the connection of the two resistors.

# generate_netlist(file_='mytestpcb1.net')


### EXAMPLE 2
from skidl import *
import sys

# Define the voltage divider module.
@SubCircuit
def vdiv(inp, outp):
    """Divide inp voltage by 3 and place it on outp net."""
    rup = Part('device', 'R', value='1K', footprint='R_0201_0603Metric')
    rlo = Part('device','R', value='500', footprint='R_0201_0603Metric')
    rup[1,2] += inp, outp
    rlo[1,2] += outp, gnd

@SubCircuit
def multi_vdiv(repeat, inp, outp):
    """Divide inp voltage by 3 ** repeat and place it on outp net."""
    for _ in range(repeat):
        out_net = Net()     # Create an output net for the current stage.
        vdiv(inp, out_net)  # Instantiate a divider stage.
        inp = out_net       # The output net becomes the input net for the next stage.
    outp += out_net         # Connect the output from the last stage to the module output net.

gnd = Net('GND')         # GLobal ground net.
input_net = Net('IN')    # Net with the voltage to be divided.
output_net = Net('OUT')  # Net with the divided voltage.
multi_vdiv(3, input_net, output_net)  # Run the input through 3 voltage dividers.

generate_netlist(file_='test2.net')
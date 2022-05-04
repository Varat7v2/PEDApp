from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

SKIDL_lib_version = '0.0.1'

skidl_examples_lib = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'R', 'dest':TEMPLATE, 'tool':SKIDL, 'description':'Resistor', 'tool_version':'kicad', 'pin':None, 'keywords':'R res resistor', 'datasheet':'~', 'value_str':'1K', '_match_pin_regex':False, 'ref_prefix':'R', 'num_units':1, 'fplist':['R_*'], 'do_erc':True, 'aliases':Alias(), 'pin':None, 'footprint':'R_0201_0603Metric', 'pins':[
            Pin(num='1',name='~',func=Pin.types.PASSIVE,do_erc=True),
            Pin(num='2',name='~',func=Pin.types.PASSIVE,do_erc=True)] })])
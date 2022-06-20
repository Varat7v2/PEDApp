from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

SKIDL_lib_version = '0.0.1'

skidl_lib = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'R', 'dest':TEMPLATE, 'tool':SKIDL, 'datasheet':'~', 'tool_version':'kicad', 'value_str':'1K', '_match_pin_regex':False, 'keywords':'R res resistor', 'description':'Resistor', 'ref_prefix':'R', 'num_units':1, 'fplist':['R_*'], 'do_erc':True, 'aliases':Alias(), 'pin':None, 'footprint':None, 'pins':[
            Pin(num='1',name='~',func=Pin.types.PASSIVE,do_erc=True),
            Pin(num='2',name='~',func=Pin.types.PASSIVE,do_erc=True)] })])
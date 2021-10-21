from pcbnew import *

board = GetBoard()

nets = board.GetNetsByName()

for netname, net in nets.items():
	print('Netcode: {}, name: {}'.format(net.GetNet(), netname))

for module in board.GetModules():
	print('Module: ', module.GetReference())


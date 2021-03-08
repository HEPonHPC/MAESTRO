import pyhenson as h
import sys
import os
import argparse
import json
import numpy as np

class SaneFormatter(argparse.RawTextHelpFormatter,
			argparse.ArgumentDefaultsHelpFormatter):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Orchestrator for the workflow',
			formatter_class=SaneFormatter)
	# parser.add_argument("--iterno", dest="ITERNO", type=int, default=2,
	# 		help="Max iteration number")
	parser.add_argument("-v", "--debug", dest="DEBUG", type=int, default=0,
			help="Turn on some debug messages")
	parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
			help="Algorithm Parameters (JSON)")

	args = parser.parse_args()
	import apprentice
	memorymap = apprentice.tools.putInMemoryMap(memoryMap=None, key="file", value=args.ALGOPARAMS)
	apprentice.tools.putInMemoryMap(memoryMap=memorymap, key="debug", value=args.DEBUG)


	# # Not used for now
	# with open (args.ALGOPARAMS, 'r') as f:
	# 	algoparamsds = json.load(f)
	# tr_radius = algoparamsds['tr']['radius']
	# # tc = TestClass()
	# # c= np.array("ss1","ss"])
	# # print(c)
	# # h.add("classss",c)
	# h.add("tr_radius",tr_radius)
	# #orc@04-03: adding array comm via henson
	# tr_center = np.array(algoparamsds['tr']['center'])
	# param_bounds = np.array(algoparamsds['param_bounds'])
	# # h.add("param_bounds",param_bounds)
	# h.add("tr_center", tr_center)

	# num_iter = args.ITERNO
	# debug = args.DEBUG
	# h.add("debug", debug)
	for k in range(apprentice.tools.getFromMemoryMap(memoryMap=memorymap,key="max_iteration")):
		apprentice.tools.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
		pyhenson = False
		try:
			h.add("MemoryMap", memorymap)
			pyhenson = True
		except:
			print(
				"Standalone` run detected. I will store data structures in files "
				"for communication between tasks")
			ds = {"MemoryMap":memorymap.tolist()}
			with open("/tmp/memorymap.json",'w') as f:
				json.dump(ds,f,indent=4)

		if apprentice.tools.getFromMemoryMap(memoryMap=memorymap,key="debug"):
			print("orchestrator: yielding to other tasks, at iter", k)
		sys.stdout.flush()
		if pyhenson: h.yield_()

	print("===terminating the workflow after", k+1, "iterations@ORCHESTRATOR===")
	sys.stdout.flush()
	os._exit(0)

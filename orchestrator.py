import sys
import os
import argparse

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

	for k in range(apprentice.tools.getFromMemoryMap(memoryMap=memorymap,key="max_iteration")):
		apprentice.tools.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
		pyhenson = False
		try:
			import pyhenson as h
			h.add("MemoryMap", memorymap)
			pyhenson = True
		except:
			apprentice.tools.writeMemoryMap(memoryMap=memorymap)

		if apprentice.tools.getFromMemoryMap(memoryMap=memorymap,key="debug"):
			print("orchestrator: yielding to other tasks, at iter", k)
		sys.stdout.flush()
		if pyhenson:
			import pyhenson as h
			h.yield_()

	print("===terminating the workflow after", k+1, "iterations@ORCHESTRATOR===")
	sys.stdout.flush()
	os._exit(0)

import sys
import os
import argparse

class SaneFormatter(argparse.RawTextHelpFormatter,
			argparse.ArgumentDefaultsHelpFormatter):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Orchestrator for the workflow',
			formatter_class=SaneFormatter)
	parser.add_argument("-v", "--debug", dest="DEBUG", default=False, action="store_true",
			help="Turn on some debug messages")
	parser.add_argument("-c", "--continue", dest="CONTINUE", default=False, action="store_true",
						help="Continue from previous iteration. Required only for local runs. "
							 "Use false in first iteration and true in subsequent iterations.")
	parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
			help="Algorithm Parameters (JSON)")

	args = parser.parse_args()
	import apprentice.tools as ato
	if args.CONTINUE:
		(memorymap, pyhenson) = ato.readMemoryMap()
		currk = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
		ato.putInMemoryMap(memoryMap=memorymap, key="debug", value=args.DEBUG)
		k = currk + 1
		ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
		pyhenson = ato.writeMemoryMap(memoryMap=memorymap)
		if ato.getFromMemoryMap(memoryMap=memorymap, key="debug"):
			print("orchestrator: yielding to other tasks, at iter", k)

	else:
		memorymap = ato.putInMemoryMap(memoryMap=None, key="file", value=args.ALGOPARAMS)
		ato.putInMemoryMap(memoryMap=memorymap, key="debug", value=args.DEBUG)
		for k in range(ato.getFromMemoryMap(memoryMap=memorymap,key="max_iteration")):
			ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
			pyhenson = ato.writeMemoryMap(memoryMap=memorymap)

			if ato.getFromMemoryMap(memoryMap=memorymap,key="debug"):
				print("orchestrator: yielding to other tasks, at iter", k)
			sys.stdout.flush()
			if pyhenson:
				from mpi4py import MPI
				comm = MPI.COMM_WORLD
				size = comm.Get_size()
				rank = comm.Get_rank()
				if rank == 0:
					print("---------------Starting Iteration {}---------------".format(k + 1))
				import pyhenson as h
				h.yield_()
			else:
				break
		if args.DEBUG:
			print("===terminating the workflow after", k+1, "iterations@ORCHESTRATOR===")
	print("---------------Starting Iteration {}---------------".format(k + 1))
	sys.stdout.flush()
	os._exit(0)

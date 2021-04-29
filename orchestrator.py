import sys
import os
import argparse

def checkStatus(memorymap):
	status = ato.getFromMemoryMap(memoryMap=memorymap, key="status")
	iterno = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
	if status > 0:
		if rank == 0 and ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel") > 0:
			print("\n===terminating the workflow with status {} : {} @ORCHESTRATOR===".format(status,ato.getStatusDef(status)))
			sys.stdout.flush()
		os._exit(status)
	else:
		if rank == 0 and "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")):
			print("--------------- Iteration {} ---------------".format(iterno + 1))
			sys.stdout.flush()

def addOutLevel(memorymap):
	ato.putInMemoryMap(memoryMap=memorymap, key="outputlevel", value=args.OUTLEVEL)
	# TEMP START
	if ato.getFromMemoryMap(memoryMap=memorymap,key="tr_radius") < 10**-5:
		ato.putInMemoryMap(memoryMap=memorymap, key="outputlevel", value=30)
	# TEMP END
	return memorymap

class SaneFormatter(argparse.RawTextHelpFormatter,
			argparse.ArgumentDefaultsHelpFormatter):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Orchestrator for the workflow',
			formatter_class=SaneFormatter)
	parser.add_argument("-o", "--outlevel", dest="OUTLEVEL", type=int, default=0,
			help="Output level")
	parser.add_argument("-c", "--continue", dest="CONTINUE", default=False, action="store_true",
						help="Continue from previous iteration. Required only for local runs. "
							 "Use false in first iteration and true in subsequent iterations.")
	parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
			help="Algorithm Parameters (JSON)")

	args = parser.parse_args()
	import apprentice.tools as ato
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	if args.CONTINUE:
		(memorymap, pyhenson) = ato.readMemoryMap()
		memorymap = addOutLevel(memorymap)
		checkStatus(memorymap)
		currk = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
		k = currk + 1
		ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
		ato.writeMemoryMap(memoryMap=memorymap)
	else:
		memorymap = ato.putInMemoryMap(memoryMap=None, key="file", value=args.ALGOPARAMS)
		memorymap = addOutLevel(memorymap)
		checkStatus(memorymap)
		for k in range(ato.getFromMemoryMap(memoryMap=memorymap,key="max_iteration")):
			ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
			memorymap = addOutLevel(memorymap)
			pyhenson = ato.writeMemoryMap(memoryMap=memorymap)
			if pyhenson:
				if rank == 0 and "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")):
					print("orchestrator: yielding to other tasks, at iter", k)
					sys.stdout.flush()
				import pyhenson as h
				h.yield_()
			else:
				break
	sys.stdout.flush()
	os._exit(0)

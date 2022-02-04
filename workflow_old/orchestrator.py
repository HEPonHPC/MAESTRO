"""
Workflow orchestrator
"""
import sys
import os
import argparse
import apprentice.tools as ato
from mpi4py import MPI

def checkStatus(memorymap,rank):
	"""
	Helper function that checks the status of the algorithm. If the algorithm is to stop
	then exit is called, otherwise the algorithm continues to the next iteration.

	:param memorymap: memory map object (see apprentice.tools)
	:param rank: MPI rank
	:type memorymap: object
	:type rank: int

	"""
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

def addOutLevel(memorymap, level):
	"""
	Update memory map with output (for print & error) level (can be updated each iteration for
	flexibility)

	:param memorymap: memory map object (see apprentice.tools)
	:param level: output level
	:type memorymap: object
	:type level: int
	:return: updated memory map
	:rtype: object

	"""
	ato.putInMemoryMap(memoryMap=memorymap, key="outputlevel", value=level)
	# TEMP START
	if ato.getFromMemoryMap(memoryMap=memorymap,key="tr_radius") < 10**-5:
		ato.putInMemoryMap(memoryMap=memorymap, key="outputlevel", value=30)
	# TEMP END
	return memorymap

def runOrchestrator():
	"""
	Run the algorithm orchestrator. Contains the logic of first vs continuing iteration, iteration update,
	creation and population of memory map, decaf henson (workflow) vs script mode run.
	"""
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	level = args.OUTLEVEL
	if args.CONTINUE:
		(memorymap, pyhenson) = ato.readMemoryMap()
		memorymap = addOutLevel(memorymap, level)
		checkStatus(memorymap,rank)
		currk = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
		k = currk + 1
		ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
		ato.writeMemoryMap(memoryMap=memorymap)
	else:
		memorymap = ato.putInMemoryMap(memoryMap=None, key="file", value=args.ALGOPARAMS)
		for k in range(ato.getFromMemoryMap(memoryMap=memorymap,key="max_iteration")):
			ato.putInMemoryMap(memoryMap=memorymap, key="iterationNo", value=k)
			memorymap = addOutLevel(memorymap, level)
			checkStatus(memorymap,rank)
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

class SaneFormatter(argparse.RawTextHelpFormatter,
			argparse.ArgumentDefaultsHelpFormatter):
	"""
	Helper class for better formatting of the script usage.
	"""
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
	runOrchestrator()


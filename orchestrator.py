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
	parser.add_argument("--iterno", dest="ITERNO", type=int, default=2,
			help="Max iteration number")
	parser.add_argument("-v", "--debug", dest="DEBUG", type=int, default=0,
			help="Turn on some debug messages")
	parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
			help="Algorithm Parameters (JSON)")

	pyhenson = True
	try:
		h.add("TestPyHenson",np.Infinity)
	except:
		print("Standalone` run detected. I will store data structures in files fon communication between tasks")
		pyhenson = False

	args = parser.parse_args()
	# Not used for now
	with open (args.ALGOPARAMS, 'r') as f:
		algoparamsds = json.load(f)
	tr_radius = algoparamsds['tr']['radius']
	# tc = TestClass()
	# c= np.array("ss1","ss"])
	# print(c)
	# h.add("classss",c)
	h.add("tr_radius",tr_radius)
	#orc@04-03: adding array comm via henson
	tr_center = np.array(algoparamsds['tr']['center'])
	param_bounds = np.array(algoparamsds['param_bounds'])
	# h.add("param_bounds",param_bounds)
	h.add("tr_center", tr_center)
	num_iter = args.ITERNO
	debug = args.DEBUG
	h.add("debug", debug)
	for k in range(num_iter):
		h.add("iter", k)
		if debug==1: print("orchestrator: yielding to other tasks, at iter", k)
		sys.stdout.flush()
		h.yield_()


	print("===terminating the workflow after", k+1, "iterations@ORCHESTRATOR===")
	sys.stdout.flush()
	os._exit(0)

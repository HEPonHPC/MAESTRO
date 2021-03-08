import pyhenson as h
import numpy as np
import json
import argparse
import sys

# Number of local minima
m = 10
# Number of parameter dimensions
d = 4
# Order of the local minima
beta = (1/m) * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
# All the local minima: d x m dimensional matrix
C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])

# https://www.sfu.ca/~ssurjano/shekel.html
def shekelObjective(x):
    outer = 0
    for i in range(m):
        bi = beta[i]
        inner = 0
        for j in range(d):
            inner += (x[j] - C[j, i]) ** 2
        outer = outer + 1 / (inner + bi)
    return -1 * outer

# https://www.sfu.ca/~ssurjano/sumpow.html
def sumOfDiffPowersObjective(x):
    sum = 0
    for ii in range(len(x)):
        xi = x[ii]
        new = (abs(xi)) ** (ii + 2)
        sum = sum + new
    return sum

def x2Objective(x):
    sum = 0
    for ii in range(len(x)):
        xi = x[ii]
        new = (ii + 1) * (xi ** 2)
        sum = sum + new
    return sum

def x2plusCObjective(x):
    return x2Objective(x) + 0.5

def x2minusCObjective(x):
    return x2Objective(x) - 0.5

def hybridObjective(x):
    return x2Objective(x) + shekelObjective(x) + sumOfDiffPowersObjective(x) +\
            x2plusCObjective(x) + x2minusCObjective(x)

def runSimulation(p,fidelity,problemname,factor=1):
    """
    Run simulation
    :param x: parameter
    :param n: Fidelity
    :return:
    """
    probFn = {
        "Shekel":shekelObjective,
        "SumOfDiffPowers":sumOfDiffPowersObjective,
        "X2":x2Objective,
        "X2plusC":x2plusCObjective,
        "X2minusC": x2minusCObjective,
        "Hybrid":hybridObjective
    }

    if problemname not in probFn: raise Exception("Problem name {} unknown".format(problemname))
    Y = np.random.normal(factor * (probFn[problemname](p)), 1 / np.sqrt(fidelity), 1)
    E = [1.]
    return Y,E

#def problem_main_program(algoparams,paramfile,binids,outfile):
def problem_main_program(algoparams,paramfile,expdata,outfile):
    #orc@15-02: generating binids from expdata
    with open(args.EXPDATA, 'r') as f:
        expdata = json.load(f)
    binids = [b for b in expdata]

    debug = h.get("debug")

    with open(algoparams,'r') as f:
        algoparamds = json.load(f)
    param_names = algoparamds["param_names"]
    fidelity = algoparamds["fidelity"]
    dim = algoparamds['dim']

    tr_center = algoparamds['tr']['center']
    parambounds = algoparamds['param_bounds'] if "param_bounds" in algoparamds and \
                                                     algoparamds['param_bounds'] is not None \
                                                    else None
    if parambounds is not None:
        for d in range(dim):
            if parambounds[d][0] > tr_center[d] or tr_center[d] > parambounds[d][1]:
                raise Exception("Starting TR center along dimension {} is not within parameter bound "
                                "[{}, {}]".format(d+1,parambounds[d][0],parambounds[d][1]))
        if debug==1: print("Phy bounds \t= {}".format(parambounds))
    else:
        if debug==1: print("Phy bounds \t= {}".format(None))

    with open(paramfile,'r') as f:
        ds = json.load(f)
    P = ds['parameters']

    HNAMES = np.array([b.split("#")[0]  for b in binids])
    FACTOR = np.array([int(b.split("#")[1])  for b in binids])
    BNAMES = binids
    PARAMS = {}
    for pno,p in enumerate(P):
        ppp = {}
        for d in range(dim):
            ppp[param_names[d]] = p[d]
        PARAMS[str(pno)] = ppp
    pnames = PARAMS[list(PARAMS.keys())[0]].keys()
    runs = sorted(list(PARAMS.keys()))
    vals = []
    errs = []
    for bno, b in enumerate(binids):
        vals.append([
            runSimulation(p,fidelity=fidelity,problemname=HNAMES[bno],factor=FACTOR[bno])[0][0]
            for p in P
        ])
        errs.append([
            runSimulation(p, fidelity=fidelity, problemname=HNAMES[bno], factor=FACTOR[bno])[1][0]
            for p in P
        ])

    if 'simulationbudgetused' not in algoparamds:
        algoparamds['simulationbudgetused'] = 0
    algoparamds['simulationbudgetused'] += fidelity * len(P)

    # print("##########")
    # print(vals)
    # print(errs)
    # print("##########")
    import h5py
    f = h5py.File(outfile, "w")

    # print(pnames)
    # print(runs)

    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'), compression=4)
    f.create_dataset("runs", data=np.char.encode(runs, encoding='utf8'), compression=4)
    pset = f.create_dataset("params", data=np.array([list(PARAMS[r].values()) for r in runs]),
                            compression=9)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]
    f.create_dataset("values", data=vals, compression=4)
    f.create_dataset("errors", data=errs, compression=4)
    f.close()

    if debug==1: print("Done. Output written to %s" % outfile)
    with open(algoparams,'w') as f:
        json.dump(algoparamds,f,indent=4)
    sys.stdout.flush()


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Simulation (with noise added)',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON)")
    #parser.add_argument("--newpin", dest="NEWPINFILE", type=str, default=None,
    #                    help="New parameters input file (JSON)")
    #parser.add_argument("-b", dest="BINIDS", type=str, default=[], nargs='+',
    #                    help="Bin ids Shekel#1 or X2#1 and so on")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-o", dest="OUTFILE", type=str, default=None,
                        help="MC Output file (HDF5)")

    args = parser.parse_args()


    k = h.get("iter") 
    MCout_1_k = "logs/MCout_1" + "_k{}.h5".format(k)
    MCout_1_kp1 = "logs/MCout_1" + "_k{}.h5".format(k + 1)
    MCout_Np_k = "logs/MCout_Np" + "_k{}.h5".format(k)

    newparams_1_kp1 = "logs/newparams_1" + "_k{}.json".format(k + 1)
    newparams_1_k = "logs/newparams_1" + "_k{}.json".format(k)
    newparams_Np_k = "logs/newparams_Np" + "_k{}.json".format(k)

    #print(args.OUTFILE)
    #sys.stdout.flush()

    #orc@25-02: getting a flag to differentiate between multiple runs or single run, basically which output file
    if args.OUTFILE =="multi":
        paramfile = newparams_Np_k 
        outfile = MCout_Np_k
    else:
        paramfile = newparams_1_kp1
        outfile = MCout_1_kp1

    gradCond = h.get("signal")
    if gradCond ==0:
        problem_main_program(
            args.ALGOPARAMS,
            paramfile,
            #args.NEWPINFILE,
            args.EXPDATA,
            #args.OUTFILE
            outfile
        )

    # print(runSimulation(P=[[4, 4, 4, 4]], fidelity=1000, problemname="Shekel"))

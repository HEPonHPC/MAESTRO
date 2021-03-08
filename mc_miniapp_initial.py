import pyhenson as h
import numpy as np
import json
import argparse
import sys
import apprentice

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


#orc@26-02:for now, putting this here, but then will be called via apprentice
def writePythiaFiles(proccardfile, pnames, points, outdir, fnamep="params.dat", fnameg="generator.cmd"):
     def readProcessCard(fname):
         with open(fname) as f:
             L = [l.strip() for l in f]
         return L
     from os.path import join, exists
     for num, p in enumerate(points):
         npad = "{}".format(num).zfill(1+int(np.ceil(np.log10(len(points)))))
         outd = join(outdir, npad)
         if not exists(outd):
             import os
             os.makedirs(outd)

         outfparams = join(outd, fnamep)
         with open(outfparams, "w") as pf:
             for k, v in zip(pnames, p):
                 pf.write("{name} {val:e}\n".format(name=k, val=v))

         outfgenerator = join(outd, fnameg)
         pc = readProcessCard(proccardfile)
         with open(outfgenerator, "w") as pg:
             for l in pc:
                 pg.write(l+"\n")
             for k, v in zip(pnames, p):
                 pg.write("{name} = {val:e}\n".format(name=k, val=v))


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
def problem_main_program(algoparams,expdata,outfile, processcard=None,outdir=None):
    #orc@15-02: generating binids from expdata
    with open(args.EXPDATA, 'r') as f:
        expdata = json.load(f)
    binids = [b for b in expdata]

    with open(algoparams,'r') as f:
        algoparamds = json.load(f)
    param_names = algoparamds["param_names"]
    fidelity = algoparamds["fidelity"]
    dim = algoparamds['dim']

    debug = h.get("debug")
    if debug==1:
        print("\n#####################################")
        print("Initially")
        print("#####################################")
        print("\Delta_1 \t= {}".format(algoparamds['tr']['radius']))
        print("N_p \t\t= {}".format(algoparamds['N_p']))
        print("dim \t\t= {}".format(dim))
        print("|B| \t\t= {}".format(len(binids)))
        print("P_1 \t\t= {}".format(algoparamds['tr']['center']))


    #orc@15-02: calculating info for paramfile.
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

    ds = {
        "parameters": [tr_center]
    }

    #orc@15-02: calculating this directly rather than reading from a file
    #with open(paramfile,'r') as f:
    #    ds = json.load(f)
    P = ds['parameters']

    #orc@26-02: emulating the py8-diy convention
    #TODO: shift to apprentice version when you have the latest version
    #apprentice.tools.writePythiaFiles(processcard,pnames,newparams,outdir)
    writePythiaFiles(processcard,param_names,[tr_center],outdir)    

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
    #parser.add_argument("-o", dest="OUTFILE", type=str, default=None,
    #                    help="MC Output file (HDF5)")
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")


    args = parser.parse_args()
    outfile = "logs/MCout_1_k0.h5" #orc@25-02: this only runs for 1 time initially so the output file name is fixed
    outdir  = "logs/pythia_1_k0"
    iter = h.get("iter")
    if iter ==0:
        problem_main_program(
            args.ALGOPARAMS,
            args.EXPDATA,
            outfile,
            args.PROCESSCARD,
            outdir
        )
    else:
        print("Skipping the initial MC run")
        sys.stdout.flush()
    # print(runSimulation(P=[[4, 4, 4, 4]], fidelity=1000, problemname="Shekel"))

import numpy as np
import json
import argparse
import sys
import apprentice.tools as ato

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
def problem_main_program(paramfile,processcard=None,memorymap = None,
                         expdatafile=None, outfile=None,outdir=None):
    with open(expdatafile, 'r') as f:
        expdata = json.load(f)
    binids = [b for b in expdata]

    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    fidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="fidelity")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")

    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")

    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    min_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                                         key="min_param_bounds")
    max_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                                         key="max_param_bounds")
    for d in range(dim):
        if min_param_bounds[d] > tr_center[d] or tr_center[d] > max_param_bounds[d]:
            raise Exception("Starting TR center along dimension {} is not within parameter bound "
                                "[{}, {}]".format(d+1,min_param_bounds[d],max_param_bounds[d]))

    with open(paramfile,'r') as f:
        ds = json.load(f)
    P = ds['parameters']
    if len(P) == 0:
        return 0

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
    xmin = []
    xmax = []
    for bno, b in enumerate(binids):
        vals.append([
            runSimulation(p,fidelity=fidelity,problemname=HNAMES[bno],factor=FACTOR[bno])[0][0]
            for p in P
        ])
        errs.append([
            runSimulation(p, fidelity=fidelity, problemname=HNAMES[bno], factor=FACTOR[bno])[1][0]
            for p in P
        ])
        xmin.append(min_param_bounds)
        xmax.append(max_param_bounds)

    # ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
    #                                 value=fidelity * len(P))
    simulationBudgetUsed = fidelity * len(P)

    import h5py
    f = h5py.File(outfile, "w")

    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'), compression=4)
    f.create_dataset("runs", data=np.char.encode(runs, encoding='utf8'), compression=4)
    pset = f.create_dataset("params", data=np.array([list(PARAMS[r].values()) for r in runs]),
                            compression=9)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]
    f.create_dataset("values", data=vals, compression=4)
    f.create_dataset("errors", data=errs, compression=4)
    f.create_dataset("xmin", data=xmin, compression=4)
    f.create_dataset("xmax", data=xmax, compression=4)
    f.close()

    if debug==1: print("mc_miniapp_simple done. Output written to %s" % outfile)
    sys.stdout.flush()
    # ato.writeMemoryMap(memoryMap=memorymap)
    return simulationBudgetUsed


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Simulation (with noise added)',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-o", dest="OPTION", type=str, default=None,
                       help="Option (initial,single, or multi)")
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")

    args = parser.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")
    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")
    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    min_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                                         key="min_param_bounds")
    max_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                                         key="max_param_bounds")

    if args.OPTION == "initial":
        paramfile = "logs/newparams_1_k0.json"
        outfile = "logs/MCout_1_k0.h5"
        outdir  = "logs/pythia_1_k0"

        simulationBudgetUsed = 0
        if k == 0 and rank==0 :
            if debug:
                with open(args.EXPDATA, 'r') as f:
                    expdata = json.load(f)
                binids = [b for b in expdata]
                print("\n#####################################")
                print("Initially")
                print("#####################################")
                print("\Delta_1 \t= {}".format(tr_radius))
                print("N_p \t\t= {}".format(N_p))
                print("dim \t\t= {}".format(dim))
                print("|B| \t\t= {}".format(len(binids)))
                print("P_1 \t\t= {}".format(tr_center))
                print("pyhenson \t= {}".format(pyhenson))
                print("Min Phy bnd \t= {}".format(min_param_bounds))
                print("Min Phy bnd \t= {}".format(max_param_bounds))

                sys.stdout.flush()

            with open (paramfile,'w') as f:
                json.dump({"parameters":[tr_center]},f,indent=4)
            ato.writePythiaFiles(args.PROCESSCARD, param_names, [tr_center], outdir) #orc@19-03: writePythiaFiles func causing problem w multiple procs

            simulationBudgetUsed = problem_main_program(
                paramfile,
                args.PROCESSCARD,
                memorymap,
                args.EXPDATA,
                outfile,
                outdir
            )
        else:
            if debug:
                print("Skipping the initial MC run since k (neq 0) = {} or rank (neq 0) = {}".format(k,rank))
                sys.stdout.flush()
        if k ==0: #orc@25-03: no need to bcast in other iterations than the first one.
            simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
            ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                               value=simulationBudgetUsed)
            ato.writeMemoryMap(memoryMap=memorymap)
    else:
        MCout_1_k = "logs/MCout_1" + "_k{}.h5".format(k)
        MCout_1_kp1 = "logs/MCout_1" + "_k{}.h5".format(k + 1)
        MCout_Np_k = "logs/MCout_Np" + "_k{}.h5".format(k)

        newparams_1_kp1 = "logs/newparams_1" + "_k{}.json".format(k + 1)
        newparams_1_k = "logs/newparams_1" + "_k{}.json".format(k)
        newparams_Np_k = "logs/newparams_Np" + "_k{}.json".format(k)

        outdir_1_kp1 = "logs/pythia_1" + "_k{}".format(k + 1)
        outdir_1_k = "logs/pythia_1" + "_k{}".format(k)
        outdir_Np_k = "logs/pythia_Np" + "_k{}".format(k)
        if args.OPTION == "multi":
            paramfile = newparams_Np_k
            outfile = MCout_Np_k
            outdir = outdir_Np_k
        else:
            paramfile = newparams_1_kp1
            outfile = MCout_1_kp1
            outdir = outdir_1_kp1


        gradCond = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientCondition")
        simulationBudgetUsed = 0
        if not gradCond and rank == 0:
            simulationBudgetUsed = problem_main_program(
                                    paramfile,
                                    args.PROCESSCARD,
                                    memorymap,
                                    args.EXPDATA,
                                    outfile,
                                    outdir
                                )
        simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
        ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                        value=simulationBudgetUsed)
        ato.writeMemoryMap(memoryMap=memorymap)

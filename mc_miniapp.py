import numpy as np
import json
import argparse
import sys
import apprentice.tools as ato
import apprentice
from subprocess import Popen, PIPE

def problem_main_program(paramfile,memorymap = None,isbebop=False,
                         outfile=None,outdir=None,pfname="params.dat"):
    size = comm.Get_size()
    rank = comm.Get_rank()
    PP = None
    if rank ==0:
        with open(paramfile, 'r') as f:
            pds = json.load(f)
        PP = pds['parameters']
    PP = comm.bcast(PP, root=0)
    if len(PP) == 0:
        return 0

    if isbebop:
        MPATH = "/home/oyildiz/mohan/mc_miniapp/pythia8rivetminiapp/miniapp"
    else:
        MPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/pythia8rivetminiapp/miniapp"

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

    import glob,os,re
    indirs = None
    if rank == 0:
        INDIRSLIST = [glob.glob(os.path.join(outdir, "*"))]
        indirs = [item for sublist in INDIRSLIST for item in sublist]
    indirs = comm.bcast(indirs, root=0)
    simulationBudgetUsed = len(indirs) * fidelity
    rankDirs = ato.chunkIt(indirs, size) if rank == 0 else None
    rankDirs = comm.scatter(rankDirs, root=0)
    re_pfname = re.compile(pfname) if pfname else None
    for num, d in enumerate(sorted(rankDirs)):
        files = glob.glob(os.path.join(d, "*"))
        param = None
        for f in files:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                param = apprentice.io.read_paramsfile(f)
        if param is None:
            raise Exception("Something went wrong. Cannot get parameter")
        pp = [param[pn] for pn in param_names]
        newloc = os.path.join(d, "ne{}_p{}.yoda".format(fidelity, str(num + 1)))
        p = Popen(
            [MPATH, str(pp[0]), str(pp[1]), str(pp[2]),
             str(fidelity), str(874673), "0", "1", newloc],
            stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        if p.returncode != 0:
            raise Exception("Running miniapp failed with return code {}".format(p.returncode))

    if debug:
        print("mc_miniapp done. Output written to %s" % outdir)
        sys.stdout.flush()
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
    parser.add_argument("-b", "--bebop", dest="BEBOP", default=False, action="store_true",
                        help="Running on BEBOP")


    args = parser.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.barrier()

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
        if k == 0:
            if debug and rank ==0:
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

            if rank ==0:
                with open(paramfile, 'w') as f:
                    json.dump({"parameters": [tr_center]}, f, indent=4)
                ato.writePythiaFiles(args.PROCESSCARD, param_names, [tr_center],
                                 outdir)

            simulationBudgetUsed = problem_main_program(
                paramfile,
                memorymap,
                args.BEBOP,
                outfile,
                outdir
            )
        else:
            if debug:
                print("Skipping the initial MC run since k (neq 0) = {} or rank (neq 0) = {}".format(k,rank))
                sys.stdout.flush()
        if k == 0:
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

        gradCond = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientCondition")

        if args.OPTION == "multi":
            paramfile = newparams_Np_k
            outfile = MCout_Np_k
            outdir = outdir_Np_k
            if not gradCond:
                simulationBudgetUsed = problem_main_program(
                    paramfile,
                    memorymap,
                    args.BEBOP,
                    outfile,
                    outdir
                )
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.writeMemoryMap(memoryMap=memorymap)
        else:
            paramfile = newparams_1_kp1
            outfile = MCout_1_kp1
            outdir = outdir_1_kp1
            simulationBudgetUsed = 0
            if not gradCond:
                if rank >= 0:
                    simulationBudgetUsed = problem_main_program(
                        paramfile,
                        memorymap,
                        args.BEBOP,
                        outfile,
                        outdir
                    )
                simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.writeMemoryMap(memoryMap=memorymap)






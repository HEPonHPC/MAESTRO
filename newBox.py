import argparse
import os
import apprentice.tools as ato
import apprentice
import h5py
import json
import numpy as np
from shutil import copyfile

def tr_update(expdatafile,wtfile):

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    kpstarfile = "logs/newparams_1" + "_k{}.json".format(k)
    kMCout = "logs/MCout_1" + "_k{}.h5".format(k)
    kp1pstarfile = "logs/newparams_1" + "_k{}.json".format(k + 1)
    kp1MCout = "logs/MCout_1" + "_k{}.h5".format(k + 1)

    valfile = "logs/valapprox" + "_k{}.json".format(k)
    errfile = "logs/errapprox" + "_k{}.json".format(k)

    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")
    gradCond = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientCondition")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")

    if debug: print("inside tr update w gradcond", gradCond)
    import sys
    sys.stdout.flush()
    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")

    with open(kpstarfile, 'r') as f:
        ds = json.load(f)
    kpstar = ds['parameters'][0]

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)
    if not gradCond:
        kDATA = apprentice.io.readH5(kMCout) #orc@19-03: parallel version doesn't like these readH5 funcs 
        idx = [i for i in range(len(kDATA))]
        with h5py.File(kMCout, "r") as f:
            tmp = f.get("index")[idx]
        mcbinids = [t.decode() for t in tmp]
        kp1DATA = apprentice.io.readH5(kp1MCout) #orc@19-03: parallel version doesn't like these readH5 funcs

        with open (kp1pstarfile,'r') as f:
            ds = json.load(f)
        kp1pstar = ds['parameters'][0]

        chi2_ra_k = IO.objective(kpstar)
        chi2_ra_kp1 = IO.objective(kp1pstar)

        chi2_mc_k = 0.
        chi2_mc_kp1 = 0.
        # print(mcbinids)
        # print(IO._binids)
        for mcnum, (_X, _Y, _E) in enumerate(kDATA):
            if mcbinids[mcnum] in IO._binids:
                ionum = IO._binids.index(mcbinids[mcnum])
                # print(_Y[0], IO._Y[ionum])
                chi2_mc_k += IO._W2[ionum]*((_Y[0]-IO._Y[ionum])**2/(_E[0]**2+IO._E[ionum]**2))
            else:
                continue

        for mcnum, (_X, _Y, _E) in enumerate(kp1DATA):
            if mcbinids[mcnum] in IO._binids:
                ionum = IO._binids.index(mcbinids[mcnum])
                # print(_Y[0], IO._Y[ionum])
                chi2_mc_kp1 += IO._W2[ionum]*((_Y[0]-IO._Y[ionum])**2/(_E[0]**2+IO._E[ionum]**2))
            else:
                continue
        # print("chi2_ra_k=\t{}\nchi2_ra_kp1=\t{}\nchi2_mc_k=\t{}\nchi2_mc_kp1=\t{}\n".format(chi2_ra_k,chi2_ra_kp1,chi2_mc_k,chi2_mc_kp1))
        if debug:
            print("chi2/ra k\t= %.2E" % (chi2_ra_k))
            print("chi2/ra k+1\t= %.2E" % (chi2_ra_kp1))
            print("chi2/mc k\t= %.2E" % (chi2_mc_k))
            print("chi2/mc k+1\t= %.2E" % (chi2_mc_kp1))

        rho = (chi2_mc_k - chi2_mc_kp1) / (chi2_ra_k - chi2_ra_kp1)
        # print("rho={}".format(rho))

        tr_eta = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_eta")
        tr_maxradius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_maxradius")

        # grad = IO.gradient(kpstar)
        if debug: print("rho k\t\t= %.3f" % (rho))
        if rho < tr_eta :
            if debug: print("rho < eta New point rejected")
            tr_radius /=2
            curr_p = kpstar
            trradmsg = "TR radius halved"
            trcentermsg = "TR center remains the same"
            copyfile(kpstarfile,kp1pstarfile)
            copyfile(kMCout,kp1MCout)
        else:
            if debug: print("rho >= eta. New point accepted")
            tr_radius = min(tr_radius*2,tr_maxradius)
            curr_p = kp1pstar
            trradmsg = "TR radius doubled"
            trcentermsg = "TR center moved to the SP amin"
    else:
        if debug: print("gradient condition failed")
        tr_radius /= 2
        curr_p = kpstar
        trradmsg = "TR radius halved"
        trcentermsg = "TR center remains the same"
        copyfile(kpstarfile,kp1pstarfile)
        copyfile(kMCout,kp1MCout)
    # put  tr_radius and curr_p in radius and center and write to algoparams
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_radius",
                       value=tr_radius)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_center",
                       value=curr_p)
    if debug: print("\Delta k+1 \t= %.2E (%s)"%(tr_radius,trradmsg))

    if debug: print("P k+1 \t\t= {} ({})".format(["%.3f"%(c) for c in curr_p],trcentermsg))

    # Stopping condition
    # get parameters
    max_iteration = ato.getFromMemoryMap(memoryMap=memorymap, key="max_iteration")
    min_gradientNorm = ato.getFromMemoryMap(memoryMap=memorymap, key="min_gradientNorm")
    max_simulationBudget = ato.getFromMemoryMap(memoryMap=memorymap, key="max_simulationBudget")

    # get budget
    simulationbudgetused = ato.getFromMemoryMap(memoryMap=memorymap, key="simulationbudgetused")

    # get gradient of model at current point
    IO._AS.setRecurrence(curr_p)
    IO._EAS.setRecurrence(curr_p)
    grad = IO.gradient(curr_p)

    status = "CONTINUE"
    if np.linalg.norm(grad) <= min_gradientNorm:
        status = 1
        if debug: print("STOP\t\t= Norm of the gradient too small {}".format(np.linalg.norm(grad)))
    elif k >= max_iteration-1:
        status = 2
        if debug: print("STOP\t\t= Max iterations reached")
    elif simulationbudgetused >= max_simulationBudget:
        status = 3
        if debug: print("STOP\t\t= Simulation budget depleted")
    else: status = 0
    if debug: print("Status\t\t= {}".format(status))
    ato.putInMemoryMap(memoryMap=memorymap, key="status",
                       value=status)
    ato.writeMemoryMap(memorymap, forceFileWrite=True)

    if status > 0:
        print("===terminating the workflow after", k+1, "iterations@TR_UPDATE===")
        sys.stdout.flush()
        os._exit(0)

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TR Update',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")

    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        tr_update(
            args.EXPDATA,
            args.WEIGHTS
        )

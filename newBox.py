import pyhenson as h
import argparse
import os
import apprentice
import h5py
import json
import numpy as np
from shutil import copyfile

def tr_update(currIterationNo,algoparams,valfile,errfile,expdatafile,wtfile,
              kpstarfile,kMCout,kp1pstarfile,kp1MCout):
    with open(algoparams, 'r') as f:
        algoparamds = json.load(f)
    #gradcond = algoparamds['tr']['gradientCondition'] #orc@15-02: converting this to henson signaling
    gradcond = h.get("signal")
    debug = h.get("debug")

    if debug==1: print("inside tr update w gradcond", gradcond)
    import sys
    sys.stdout.flush()
    tr_radius = algoparamds['tr']['radius']

    if currIterationNo!=0:
        with open(kpstarfile, 'r') as f:
            ds = json.load(f)
        kpstar = ds['parameters'][0]
    else:
        tr_center = algoparamds['tr']['center']

        dim = algoparamds['dim']
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

        with open(kpstarfile, 'w') as f:
            json.dump(ds, f, indent=4)

        kpstar = ds['parameters'][0]

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)
    #if gradcond=="NO":
    if gradcond==0:
        kDATA = apprentice.io.readH5(kMCout)
        idx = [i for i in range(len(kDATA))]
        with h5py.File(kMCout, "r") as f:
            tmp = f.get("index")[idx]
        mcbinids = [t.decode() for t in tmp]
        kp1DATA = apprentice.io.readH5(kp1MCout)

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
        if debug==1:
            print("chi2/ra k\t= %.2E" % (chi2_ra_k))
            print("chi2/ra k+1\t= %.2E" % (chi2_ra_kp1))
            print("chi2/mc k\t= %.2E" % (chi2_mc_k))
            print("chi2/mc k+1\t= %.2E" % (chi2_mc_kp1))

        rho = (chi2_mc_k - chi2_mc_kp1) / (chi2_ra_k - chi2_ra_kp1)
        # print("rho={}".format(rho))

        eta = algoparamds['tr']['eta']
        sigma = algoparamds['tr']['sigma']
        tr_maxradius = algoparamds['tr']['maxradius']

        # grad = IO.gradient(kpstar)
        if debug==1: print("rho k\t\t= %.3f" % (rho))
        if rho < eta :
                # or np.linalg.norm(grad) <= sigma * tr_radius:
            if debug==1: print("rho < eta New point rejected")
            tr_radius /=2
            curr_p = kpstar
            trradmsg = "TR radius halved"
            trcentermsg = "TR center remains the same"
            copyfile(kpstarfile,kp1pstarfile)
            copyfile(kMCout,kp1MCout)
        else:
            if debug==1: print("rho >= eta. New point accepted")
            tr_radius = min(tr_radius*2,tr_maxradius)
            curr_p = kp1pstar
            # copyfile(src, dst)
            # copyfile(kp1pstarfile,kpstarfile)
            # copyfile(kp1MCout,kMCout)
            trradmsg = "TR radius doubled"
            trcentermsg = "TR center moved to the SP amin"
    else:
        if debug==1: print("gradient condition failed")
        tr_radius /= 2
        curr_p = kpstar
        trradmsg = "TR radius halved"
        trcentermsg = "TR center remains the same"
        copyfile(kpstarfile,kp1pstarfile)
        copyfile(kMCout,kp1MCout)
    # put  tr_radius and curr_p in radius and center and write to algoparams
    algoparamds['tr']['radius'] = tr_radius
    algoparamds['tr']['center'] = curr_p
    if debug==1: print("\Delta k+1 \t= %.2E (%s)"%(tr_radius,trradmsg))

    if debug==1: print("P k+1 \t\t= {} ({})".format(["%.3f"%(c) for c in curr_p],trcentermsg))

    # Stopping condition
    # get parameters
    max_iteration = algoparamds['max_iteration']
    min_gradientNorm = algoparamds['min_gradientNorm']
    max_simulationBudget = algoparamds['max_simulationBudget']

    # get budget
    simulationbudgetused = algoparamds['simulationbudgetused']

    # get gradient of model at current point
    IO._AS.setRecurrence(curr_p)
    IO._EAS.setRecurrence(curr_p)
    grad = IO.gradient(curr_p)

    status = "CONTINUE"
    if np.linalg.norm(grad) <= min_gradientNorm:
        status = "STOP"
        if debug==1: print("STOP\t\t= Norm of the gradient too small {}".format(np.linalg.norm(grad)))
    if currIterationNo >= max_iteration-1:
        status = "STOP"
        if debug==1: print("STOP\t\t= Max iterations reached")
    if simulationbudgetused >= max_simulationBudget:
        status = "STOP"
        if debug==1: print("STOP\t\t= Simulation budget depleted")
    if debug==1: print(status)
    algoparamds['status'] = status
    with open(algoparams,'w') as f:
        json.dump(algoparamds,f,indent=4)

    if status == "STOP":
        print("===terminating the workflow after", currIterationNo+1, "iterations@TR_UPDATE===")
        sys.stdout.flush()
        os._exit(0)

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TR Update',
                                     formatter_class=SaneFormatter)
    #parser.add_argument("--iterno", dest="ITERNO", type=int, default=0,
    #                    help="Current iteration number")
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON)")
    #parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
    #                    help="Value approximation file name (JSON)")
    #parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
    #                    help="Error approximation file name (JSON)")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")
    #parser.add_argument("--kpstarfile", dest="KPSTARFILE", type=str, default=None,
    #                    help="p^* parameter file from iteration k (JSON)")
    #parser.add_argument("--kp1pstarfile", dest="KP1PSTARFILE", type=str, default=None,
    #                    help="p^* parameter outfile from iteration k + 1 (JSON)")
    #parser.add_argument("--kMCout", dest="KMCOUT", type=str, default=None,
    #                    help="MC OUT (HDF5) from iteration k")
    #parser.add_argument("--kp1MCout", dest="KP1MCOUT", type=str, default=None,
    #                    help="MC OUT (HDF5) from iteration k+1")

    k = h.get("iter")
    MCout_1_k = "logs/MCout_1" + "_k{}.h5".format(k)
    MCout_1_kp1 = "logs/MCout_1" + "_k{}.h5".format(k + 1)
    
    newparams_1_kp1 = "logs/newparams_1" + "_k{}.json".format(k + 1)
    newparams_1_k = "logs/newparams_1" + "_k{}.json".format(k)

    valapproxfile_k = "logs/valapprox" + "_k{}.json".format(k)
    errapproxfile_k = "logs/errapprox" + "_k{}.json".format(k)

    args = parser.parse_args()
    tr_update(
        k,
#        args.ITERNO,
        args.ALGOPARAMS,
#        args.VALAPPFILE,
#        args.ERRAPPFILE,
        valapproxfile_k,
        errapproxfile_k,
        args.EXPDATA,
        args.WEIGHTS,
        newparams_1_k,
#        args.KPSTARFILE,
#        args.KMCOUT,
        MCout_1_k,
        newparams_1_kp1,
#        args.KP1PSTARFILE,
        MCout_1_kp1
#        args.KP1MCOUT
    )

import argparse
import os
import apprentice.tools as ato
import apprentice
import h5py
import json
import numpy as np
import shutil,errno

def isclose(a, b, rel_tol=1e-03, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
def getInfNorm(param):
    distarr = [np.abs(p) for p in param]
    infn = max(distarr)
    return infn

def copyanything(src, dst):
    try:
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def tr_update(memorymap,expdatafile,wtfile):
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    kpstarfile = "logs/newparams_1" + "_k{}.json".format(k)
    kMCoutH5 = "logs/MCout_1" + "_k{}.h5".format(k)
    kMCoutYODA = "logs/pythia_1" + "_k{}".format(k)
    kp1pstarfile = "logs/newparams_1" + "_k{}.json".format(k + 1)
    kp1MCoutH5 = "logs/MCout_1" + "_k{}.h5".format(k + 1)
    kp1MCoutYODA = "logs/pythia_1" + "_k{}".format(k + 1)

    valfile = "logs/valapprox" + "_k{}.json".format(k)
    errfile = "logs/errapprox" + "_k{}.json".format(k)

    oloptions = ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel"))
    debug = True if "All" in oloptions else False

    gradCond = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientCondition")
    status = ato.getFromMemoryMap(memoryMap=memorymap, key="status")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    currIteration = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    no_iters_at_max_fidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="no_iters_at_max_fidelity")
    radius_at_which_max_fidelity_reached = ato.getFromMemoryMap(memoryMap=memorymap, key="radius_at_which_max_fidelity_reached")

    import sys
    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")

    if status == 0:
        with open(kpstarfile, 'r') as f:
            ds = json.load(f)
        kpstar = ds['parameters'][0]

        IO = apprentice.appset.TuningObjective2(wtfile,
                                                expdatafile,
                                                valfile,
                                                errfile)

        if debug: print("inside tr update w gradcond", gradCond)
        sys.stdout.flush()
        fidelityused = None
        old_tr_radius = None
        if not gradCond:
            mcbinids = IO._binids
            if ato.getFromMemoryMap(memoryMap=memorymap, key="useYODAoutput"):
                # kDATA,binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
                #     [kMCoutYODA],"params.dat",wtfile)
                # kp1DATA,binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
                #     [kp1MCoutYODA], "params.dat", wtfile)
                import glob
                INDIRSLIST = glob.glob(os.path.join(kMCoutYODA, "*"))
                kDATA = apprentice.io.readSingleYODAFile(
                    INDIRSLIST[0],"params.dat",wtfile)

                INDIRSLIST = glob.glob(os.path.join(kp1MCoutYODA, "*"))
                kp1DATA = apprentice.io.readSingleYODAFile(
                    INDIRSLIST[0], "params.dat", wtfile)
            else:
                kDATA = apprentice.io.readH5(kMCoutH5)
                kp1DATA = apprentice.io.readH5(kp1MCoutH5)

            with open (kp1pstarfile,'r') as f:
                ds = json.load(f)
            kp1pstar = ds['parameters'][0]
            fidelityused = ds['at fidelity'][0]
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
                print("chi2/ra k\t= %.4E" % (chi2_ra_k))
                print("chi2/ra k+1\t= %.4E" % (chi2_ra_kp1))
                print("chi2/mc k\t= %.4E" % (chi2_mc_k))
                print("chi2/mc k+1\t= %.4E" % (chi2_mc_kp1))

            rho = (chi2_mc_k - chi2_mc_kp1) / (chi2_ra_k - chi2_ra_kp1)
            # print("rho={}".format(rho))

            tr_eta = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_eta")
            tr_maxradius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_maxradius")

            # grad = IO.gradient(kpstar)
            if debug: print("rho k\t\t= %.4E" % (rho))
            normOfStep = getInfNorm(np.array(kp1pstar)-np.array(tr_center))
            if rho < tr_eta :
                if debug: print("rho < eta New point rejected")
                tr_radius = min(tr_radius,normOfStep)/2
                curr_p = kpstar
                trradmsg = "TR radius halved"
                trcentermsg = "TR center remains the same"
                trcenterstatus = "R"
                copyanything(kpstarfile,kp1pstarfile)
                if ato.getFromMemoryMap(memoryMap=memorymap, key="useYODAoutput"):
                    copyanything(kMCoutYODA, kp1MCoutYODA)
                else:
                    copyanything(kMCoutH5,kp1MCoutH5)
            else:
                if debug: print("rho >= eta. New point accepted")
                if isclose(getInfNorm(np.array(kp1pstar)-np.array(tr_center)),tr_radius):
                    tr_radius = min(tr_radius*2,tr_maxradius)
                    trradmsg = "TR radius doubled"
                    trcenterstatus = "A"
                else:
                    trradmsg = "TR radius stays the same"
                    trcenterstatus = "M"
                curr_p = kp1pstar
                trcentermsg = "TR center moved to the SP amin"
            if rank==0 and "1lineoutput" in oloptions:
                pgnorm = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientNorm")
                old_tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
                str = ""
                if currIteration %10 == 0:
                    str = "iter\tGC   PGNorm     \Delta_k" \
                          "     NormOfStep  S   C_RA(P_k)  C_RA(P_{k+1}) C_MC(P_k)  C_MC(P_{k+1}) N_e(apprx)    \\rho\n"
                str += "%d\tF %.6E %.6E %.6E %s %.6E %.6E %.6E %.6E %.4E %.6E"\
                      %(currIteration+1,pgnorm,old_tr_radius,normOfStep,trcenterstatus,chi2_ra_k,chi2_ra_kp1,chi2_mc_k,chi2_mc_kp1,fidelityused,rho)
                print(str)
        else:
            fidelityused = None
            if debug: print("gradient condition failed")
            tr_radius /= 2
            curr_p = kpstar
            trradmsg = "TR radius halved"
            trcentermsg = "TR center remains the same"
            trcenterstatus = "R"
            copyanything(kpstarfile,kp1pstarfile)
            if ato.getFromMemoryMap(memoryMap=memorymap, key="useYODAoutput"):
                copyanything(kMCoutYODA, kp1MCoutYODA)
            else:
                copyanything(kMCoutH5,kp1MCoutH5)
            if rank==0 and "1lineoutput" in oloptions:
                pgnorm = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientNorm")
                old_tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
                str = ""
                if currIteration %10 == 0:
                    str = "iter\tGC   PGNorm     \Delta_k" \
                          "     NormOfStep  S   C_RA(P_k)  C_RA(P_{k+1}) C_MC(P_k)  C_MC(P_{k+1}) N_e(apprx)    \\rho\n"
                normOfStep = 0.
                str += "%d\tT %.6E %.6E %.6E %s" \
                       %(currIteration+1,pgnorm,old_tr_radius,normOfStep,trcenterstatus)
                print(str)
        # put  tr_radius and curr_p in radius and center and write to algoparams
        # ato.putInMemoryMap(memoryMap=memorymap, key="tr_radius",
        #                    value=tr_radius)
        # ato.putInMemoryMap(memoryMap=memorymap, key="tr_center",
        #                    value=curr_p)
        if debug: print("\Delta k+1 \t= %.4E (%s)"%(tr_radius,trradmsg))

        if debug or "PKp1" in oloptions: print("P k+1 \t\t= {} ({})".format(["%.4f"%(c) for c in curr_p],trcentermsg))

        if "NormOfStep" in oloptions:
            normofstep = np.linalg.norm(np.array(curr_p)-np.array(tr_center))
            print("Norm of Step \t= %.8E (%s)"%(normofstep,trcentermsg))

        # Stopping condition
        # get parameters
        max_iteration = ato.getFromMemoryMap(memoryMap=memorymap, key="max_iteration")
        max_simulationBudget = ato.getFromMemoryMap(memoryMap=memorymap, key="max_simulationBudget")
        maxfidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="maxfidelity")
        max_fidelity_iteration = ato.getFromMemoryMap(memoryMap=memorymap, key="max_fidelity_iteration")

        # get budget/current metrics
        simulationbudgetused = ato.getFromMemoryMap(memoryMap=memorymap, key="simulationbudgetused")

        if not gradCond and fidelityused is not None:
            if fidelityused >= maxfidelity:
                no_iters_at_max_fidelity += 1
                if radius_at_which_max_fidelity_reached == 0:
                    radius_at_which_max_fidelity_reached = old_tr_radius
            else:
                no_iters_at_max_fidelity = 0

        def orderOfMagnitude(number):
            return np.floor(np.log10(number))

        if k >= max_iteration-1:
            status = 2
        elif simulationbudgetused >= max_simulationBudget:
            status = 3
        elif radius_at_which_max_fidelity_reached > 0 and orderOfMagnitude(old_tr_radius) <= orderOfMagnitude(radius_at_which_max_fidelity_reached)-2:
            status = 5
        elif no_iters_at_max_fidelity >= max_fidelity_iteration:
            status = 6
        else: status = 0
        if debug: print("Status\t\t= {} : {}".format(status,ato.getStatusDef(status)))

        return (status,tr_radius,curr_p,no_iters_at_max_fidelity,radius_at_which_max_fidelity_reached)
    else:
        return (status,tr_radius,tr_center,no_iters_at_max_fidelity,radius_at_which_max_fidelity_reached)

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

    comm.barrier()
    (memorymap, pyhenson) = ato.readMemoryMap()
    status,radius,center,no_iters_at_max_fidelity,radius_at_which_max_fidelity_reached = None,None,None,None,None
    if rank == 0:
        (status,radius,center,no_iters_at_max_fidelity,radius_at_which_max_fidelity_reached) = tr_update(
            memorymap,
            args.EXPDATA,
            args.WEIGHTS
        )
    status = comm.bcast(status, root=0)
    radius = comm.bcast(radius, root=0)
    center = comm.bcast(center, root=0)
    no_iters_at_max_fidelity = comm.bcast(no_iters_at_max_fidelity, root=0)
    radius_at_which_max_fidelity_reached = comm.bcast(radius_at_which_max_fidelity_reached, root=0)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_radius",
                       value=radius)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_center",
                       value=center)
    ato.putInMemoryMap(memoryMap=memorymap, key="status",
                       value=status)
    ato.putInMemoryMap(memoryMap=memorymap, key="no_iters_at_max_fidelity",
                       value=no_iters_at_max_fidelity)
    ato.putInMemoryMap(memoryMap=memorymap, key="radius_at_which_max_fidelity_reached",
                       value=radius_at_which_max_fidelity_reached)
    ato.writeMemoryMap(memorymap, forceFileWrite=True)
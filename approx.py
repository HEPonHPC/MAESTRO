import h5py
import apprentice
import json
import sys,os
import argparse
import numpy as np
import apprentice.tools as ato

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
def run_approx(memorymap,prevparamfile,valoutfile,
               erroutfile,expdatafile,wtfile):
    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    currIteration = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception as e:
        print("Exception when trying to import mpi4py:", e)
        comm = None
        pass

    if debug: print("Starting approximation --")
    # print("CHANGE ME TO THE PARALLEL VERSION")
    # assert (erroutfile != interpolationdatafile)
    # assert (valoutfile != interpolationdatafile)
    with open (prevparamfile,'r') as f:
        prevparamds = json.load(f)

    Xtouse,Ytouse,Etouse = [],[],[]
    if len(prevparamds["parameters"]) > 0:
        k_ptype_done = []
        for pno,param in enumerate(prevparamds["parameters"]):
            pnoToRead = []
            k = prevparamds["{}".format(pno)]["k"]
            ptype = prevparamds["{}".format(pno)]["ptype"]
            k_ptype_str = "k{}_ptype{}".format(k,ptype)
            if k_ptype_str in k_ptype_done:
                continue

            pnoToRead.append(pno)

            for pnoinner in range(pno+1,len(prevparamds["parameters"])):
                kinner = prevparamds["{}".format(pno)]["k"]
                ptypeinner = prevparamds["{}".format(pnoinner)]["ptype"]
                k_ptype_str_inner = "k{}_ptype{}".format(kinner, ptypeinner)
                if k_ptype_str_inner in k_ptype_done:
                    continue
                if k_ptype_str == k_ptype_str_inner:
                    pnoToRead.append(pnoinner)

            k_ptype_done.append(k_ptype_str)

            if ato.getFromMemoryMap(memoryMap=memorymap, key="useYODAoutput"):
                # YODA directory parsing here
                MCoutprev = "logs/pythia_{}_k{}".format(ptype,k)
                DATAprev, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
                    [MCoutprev], "params.dat",wtfile,
                    storeAsH5=None, comm=comm)
            else:
                MCoutprev = "logs/MCout_{}_k{}.h5".format(ptype, k)
                DATAprev, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataH5(
                    MCoutprev, wtfile, comm=comm)

            X,Y,E = DATAprev[0]

            indexToRead = []
            for no in pnoToRead:
                indexToRead.append(next(i for i, _ in enumerate(X)
                                        if np.all(np.isclose(_, prevparamds["parameters"][no]))))
            if len(Xtouse) == 0:
                for num in range(len(DATAprev)):
                    Xtouse.append([])
                    Ytouse.append([])
                    Etouse.append([])

            for num, (X, Y, E) in  enumerate(DATAprev):
                for index in indexToRead:
                    Xtouse[num].append(X[index])
                    Ytouse[num].append(Y[index])
                    Etouse[num].append(E[index])

    if len(prevparamds["parameters"]) < N_p:
        if ato.getFromMemoryMap(memoryMap=memorymap, key="useYODAoutput"):
            # YODA directory parsing here
            interpolationdatadir = "logs/pythia_Np" + "_k{}".format(currIteration)
            DATAnew, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
                [interpolationdatadir], "params.dat",
                wtfile, storeAsH5=None, comm=comm)
        else:
            interpolationdatafile = "logs/MCout_Np" + "_k{}.h5".format(currIteration)
            DATAnew, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataH5(
                interpolationdatafile, wtfile, comm=comm)

        if len(Xtouse) == 0:
            for num in range(len(DATAnew)):
                Xtouse.append([])
                Ytouse.append([])
                Etouse.append([])

        for num, (X, Y, E) in enumerate(DATAnew):
            for x in X:
                Xtouse[num].append(x)
            for y in Y:
                Ytouse[num].append(y)
            for e in E:
                Etouse[num].append(e)

    comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    if debug: print("[{}] will proceed to calculate approximations for {} objects".format(rank, len(Ytouse)))
    sys.stdout.flush()

    # idx = [i for i in range(len(DATA))]
    valapp = {}
    errapp = {}

    import time
    t4 = time.time()
    import datetime

    # S = apprentice.Scaler(DATA[0][0])  # Let's assume that all X are the same for simplicity
    # print("Halfway reporting: before generating the output file --")
    for num, (X, Y, E) in  enumerate(zip(Xtouse, Ytouse, Etouse)):
        thisBinId = binids[num]
        if debug:
            if rank == 0 or rank == size - 1:
                if ((num + 1) % 5 == 0):
                    now = time.time()
                    tel = now - t4
                    ttg = tel * (len(Ytouse) - num) / (num + 1)
                    eta = now + ttg
                    eta = datetime.datetime.fromtimestamp(now + ttg)
                    sys.stdout.write(
                        "{}[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(
                            80 * " " if rank > 0 else "", rank, num + 1, len(Ytouse), tel, ttg,
                            eta.strftime('%Y-%m-%d %H:%M:%S')), )
                    sys.stdout.flush()
        # print(_X)
        # print(_Y)
        try:
            # print("\n\n\n\n")
            # print(_X,_Y,_E)
            val = apprentice.RationalApproximation(X, Y, order=(2, 0), pnames=pnames)
            # val._vmin = val.fmin(nsamples=100,nrestart=20)
            # val._vmax = val.fmax(nsamples=100, nrestart=20)
            val._xmin = xmin[num]
            val._xmax = xmax[num]

            err = apprentice.RationalApproximation(X, E, order=(1, 0), pnames=pnames)
        except AssertionError as error:
            print(error)
        valapp[thisBinId] = val.asDict
        errapp[thisBinId] = err.asDict
    ALLVALAPP = comm.gather(valapp, root=0)
    ALLERRAPP = comm.gather(errapp, root=0)
    t5 = time.time()
    gradCondToWrite = False
    if rank == 0:
        print()
        if debug: print("Approximation calculation took {} seconds".format(t5 - t4))
        sys.stdout.flush()


        # What do these do? Are the next 4 lines required
        # S.save("{}.scaler".format(valoutfile))
        # S.save("{}.scaler".format(erroutfile))
        # S.save(valoutfile)
        # S.save(erroutfile)


        # JD = {x.decode(): y.asDict for x, y in zip(binids, valapp)}
        from collections import OrderedDict
        JD = OrderedDict()
        a = {}
        for apps in ALLVALAPP:
            a.update(apps)
        for k in a.keys():
            JD[k] = a[k]
        with open(valoutfile, "w") as f:
            json.dump(JD, f,indent=4)

        JD = OrderedDict()
        a = {}
        for apps in ALLERRAPP:
            a.update(apps)
        for k in a.keys():
            JD[k] = a[k]
        with open(erroutfile, "w") as f:
            json.dump(JD, f,indent=4)

        # print("Done --- approximation of {} objects written to {} and {}".format(
        #         len(idx), valoutfile, erroutfile))

        tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
        tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
        tr_sigma = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_sigma")


        #print("BYE from approx")
        #sys.stdout.flush()
        try:
            IO = apprentice.appset.TuningObjective2(wtfile,
                                                    expdatafile,
                                                    valoutfile,
                                                    erroutfile, debug=debug)
            IO._AS.setRecurrence(tr_radius)
            IO._EAS.setRecurrence(tr_radius)
            grad = IO.gradient(tr_center)
            if np.linalg.norm(grad) <= tr_sigma * tr_radius:
                # ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                #                    value=True)
                gradCondToWrite = True
            else:
                gradCondToWrite = False
                # ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                #                value=False)
            if debug: print(
                "||grad|| \t= %.3f <=> %.3f" % (np.linalg.norm(grad), tr_sigma * tr_radius))
        except:
            # ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
            #                    value=False)
            gradCondToWrite = False
            pass

        sys.stdout.flush()
    gradCondToWrite = comm.bcast(gradCondToWrite, root=0)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                                          value=gradCondToWrite)
    ato.writeMemoryMap(memorymap)
    #comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    print("BYE from approx", rank, gradCondToWrite)
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct Model',
                                     formatter_class=SaneFormatter)
    # parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
    #                     help="Algorithm Parameters (JSON)")
#    parser.add_argument("-i", dest="INTERPOLATIONDATAFILE", type=str, default=None,
#                        help="Interpolation data (MC HDF5) file")
#    parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
#                        help="Value approximation output file name (JSON)")
#    parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
#                        help="Error approximation output file name (JSON)")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")

    args = parser.parse_args()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")

    MCout_Np_k = "logs/MCout_Np" + "_k{}.h5".format(k)
    valapproxfile_k = "logs/valapprox" + "_k{}.json".format(k)
    errapproxfile_k = "logs/errapprox" + "_k{}.json".format(k)
    prevparams_Np_k = "logs/prevparams_Np" + "_k{}.json".format(k)

    run_approx(
        memorymap,
        prevparams_Np_k,
        valapproxfile_k,
        errapproxfile_k,
        args.EXPDATA,
        args.WEIGHTS
    )

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
def run_approx(memorymap,interpolationdatafile,valoutfile, erroutfile,expdatafile,wtfile):
    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception as e:
        print("Exception when trying to import mpi4py:", e)
        comm = None
        pass

    comm.barrier()
    if debug: print("Starting approximation --")
    # print("CHANGE ME TO THE PARALLEL VERSION")
    assert (erroutfile != interpolationdatafile)
    assert (valoutfile != interpolationdatafile)

    #orc@19-03: os.path.isdir not good for multi procs
    DATA, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataH5(interpolationdatafile, wtfile,comm=comm)

    # DATA = apprentice.io.readH5(interpolationdatafile)
    # print(DATA)
    # pnames = apprentice.io.readPnamesH5(interpolationdatafile, xfield="params")
    #if os.path.isfile(interpolationdatafile):
    #    DATA, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataH5(interpolationdatafile, wtfile,comm=comm)
    #elif os.path.isdir(interpolationdatafile):
        # YODA directory parsing here
    #    DATA, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(interpolationdatafile, "params.dat",
    #                                                                        wtfile, storeAsH5=False,comm=comm)
    #else:
    #    print("{} neither directory nor file, exiting".format(args[0]))
    #    exit(1)

    comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    if debug: print("[{}] will proceed to calculate approximations for {} objects".format(rank, len(DATA)))
    sys.stdout.flush()

    # idx = [i for i in range(len(DATA))]
    valapp = {}
    errapp = {}

    import time
    t4 = time.time()
    import datetime

    # S = apprentice.Scaler(DATA[0][0])  # Let's assume that all X are the same for simplicity
    # print("Halfway reporting: before generating the output file --")
    for num, (X, Y, E) in  enumerate(DATA):
        thisBinId = binids[num]
        if debug:
            if rank == 0 or rank == size - 1:
                if ((num + 1) % 5 == 0):
                    now = time.time()
                    tel = now - t4
                    ttg = tel * (len(DATA) - num) / (num + 1)
                    eta = now + ttg
                    eta = datetime.datetime.fromtimestamp(now + ttg)
                    sys.stdout.write(
                        "{}[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(
                            80 * " " if rank > 0 else "", rank, num + 1, len(DATA), tel, ttg,
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
            json.dump(JD, f)

        JD = OrderedDict()
        a = {}
        for apps in ALLERRAPP:
            a.update(apps)
        for k in a.keys():
            JD[k] = a[k]
        with open(erroutfile, "w") as f:
            json.dump(JD, f)

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
    comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    print("BYE from approx")
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
    run_approx(
        memorymap,
        MCout_Np_k,
        valapproxfile_k,
        errapproxfile_k,
        args.EXPDATA,
        args.WEIGHTS
    )

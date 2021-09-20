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

def projection(X,MIN,MAX):
    return np.array([
        min(max(x,mi),ma) for x,mi,ma in zip(X,MIN,MAX)
    ])

def run_approx(memorymap,expdatafile,wtfile):
    oloptions = ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel"))
    debug = True if "All" in oloptions else False
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")

    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    currIteration = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    min_gradientNorm = ato.getFromMemoryMap(memoryMap=memorymap, key="min_gradientNorm")
    max_fidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="maxfidelity")

    valoutfile = "logs/valapprox" + "_k{}.json".format(currIteration)
    erroutfile = "logs/errapprox" + "_k{}.json".format(currIteration)
    valscaledoutfile = "logs/valapproxscaled" + "_k{}.json".format(currIteration)
    errscaledoutfile = "logs/errapproxscaled" + "_k{}.json".format(currIteration)
    prevparamfile = "logs/prevparams_Np" + "_k{}.json".format(currIteration)
    functionvaloutfile = "logs/functionvalues_Np" + "_k{}.json".format(currIteration)
    kpstarfile = "logs/newparams_1" + "_k{}.json".format(currIteration)

    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    tr_sigma = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_sigma")

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
                kinner = prevparamds["{}".format(pnoinner)]["k"]
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

            X,Y,E = (None,None,None)
            for num, (Xloc, Yloc, Eloc) in  enumerate(DATAprev):
                if len(Yloc)>0:
                    X,Y,E = DATAprev[num]
                    break
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
                    if len(Y)>0: Xtouse[num].append(X[index])
                    if len(Y)>0: Ytouse[num].append(Y[index])
                    if len(Y)>0: Etouse[num].append(E[index])

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
                if len(Y)>0:Xtouse[num].append(x)
            for y in Y:
                if len(Y)>0:Ytouse[num].append(y)
            for e in E:
                if len(Y)>0:Etouse[num].append(e)

    comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    if debug: print("[{}] will proceed to calculate approximations for {} objects".format(rank, len(Ytouse)))
    sys.stdout.flush()

    # idx = [i for i in range(len(DATA))]
    valapp = {}
    errapp = {}
    valappscaled = {}
    errappscaled = {}

    import time
    t4 = time.time()
    import datetime

    # S = apprentice.Scaler(DATA[0][0])  # Let's assume that all X are the same for simplicity
    # print("Halfway reporting: before generating the output file --")
    if rank==0 and "interpolationPoints" in oloptions:
        print("########################")
        print("Interpolation Points")
        print("########################")
        for P in Xtouse[0]:
            str = ""
            for p in P:
                str += "%.3E\t"%p
            print(str)
    gradCondToWrite = False
    pgradnorm = 1.0
    statusToWrite = 0
    if rank == 0:
        for num, (X, Y, E) in  enumerate(zip(Xtouse, Ytouse, Etouse)):
            if len(Y) < apprentice.tools.numCoeffsRapp(dim,(1,0)):
                statusToWrite = 7
                break
    statusToWrite = comm.bcast(statusToWrite, root=0)
    if statusToWrite == 0:
        Sclocal = apprentice.Scaler(Xtouse[0],pnames=pnames)
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
                order = [3,0]
                if tr_radius < 10**-1:
                    order = [2,0]
                if tr_radius < 10**-3:
                    order = [1,0]
                while len(Y) < apprentice.tools.numCoeffsRapp(dim,tuple(order)):
                    order[0] -= 1
                order = tuple(order)
                Xscaled = [Sclocal.scale(x) for x in X]
                val = apprentice.RationalApproximation(X, Y, order=order, pnames=pnames)
                val._xmin = xmin[num]
                val._xmax = xmax[num]
                err = apprentice.RationalApproximation(X, E, order=(1, 0), pnames=pnames)

                valscaled = apprentice.RationalApproximation(Xscaled, Y, order=(1,0), pnames=pnames)
                errscaled = apprentice.RationalApproximation(Xscaled, E, order=(1, 0), pnames=pnames)
            except AssertionError as error:
                print(error)
            valapp[thisBinId] = val.asDict
            errapp[thisBinId] = err.asDict
            valappscaled[thisBinId] = valscaled.asDict
            errappscaled[thisBinId] = errscaled.asDict
        ALLVALAPP = comm.gather(valapp, root=0)
        ALLERRAPP = comm.gather(errapp, root=0)
        ALLVALAPPSCALED = comm.gather(valappscaled, root=0)
        ALLERRAPPSCALED = comm.gather(errappscaled, root=0)
        t5 = time.time()
        if rank == 0:
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

            JD = OrderedDict()
            a = {}
            for apps in ALLVALAPPSCALED:
                a.update(apps)
            for k in a.keys():
                JD[k] = a[k]
            with open(valscaledoutfile, "w") as f:
                json.dump(JD, f,indent=4)

            JD = OrderedDict()
            a = {}
            for apps in ALLERRAPPSCALED:
                a.update(apps)
            for k in a.keys():
                JD[k] = a[k]
            with open(errscaledoutfile, "w") as f:
                json.dump(JD, f,indent=4)

            if rank==0 and "MC_RA_functionValue" in oloptions:
                str = ""
                for num, (X, Y, E) in  enumerate(zip(Xtouse, Ytouse, Etouse)):
                    thisBinId = binids[num]
                    str += "########################\n{}\n########################\n\n".format(thisBinId)
                    str += "P(dim = {})\t\t\tMC(P)\tr_v(P)\t\t\Delta MC(P)\tr_e(P)\n".format(dim)
                    for pno,P in enumerate(X):
                        for p in P:
                            str += "%.2E\t"%(p)
                        str += "\t"
                        str += "%.2E\t"%(Y[pno])
                        str += "%.2E\t"%(apprentice.RationalApproximation(initDict=valapp[thisBinId])(P))
                        str += "\t"
                        str += "%.2E\t"%(E[pno])
                        str += "%.2E\t"%(apprentice.RationalApproximation(initDict=errapp[thisBinId])(P))
                        str += "\n"
                    str+="\n\n"
                with open(functionvaloutfile, 'w') as f:
                    print(str, file=f)

            # print("Done --- approximation of {} objects written to {} and {}".format(
            #         len(idx), valoutfile, erroutfile))



            #print("BYE from approx")
            #sys.stdout.flush()
            try:

                IO = apprentice.appset.TuningObjective2(wtfile,
                                                        expdatafile,
                                                        valscaledoutfile,
                                                        errscaledoutfile,
                                                        debug=debug)
                tr_center_scaled = Sclocal.scale(tr_center)
                IO._AS.setRecurrence(tr_center_scaled)
                IO._EAS.setRecurrence(tr_center_scaled)
                grad = IO.gradient(tr_center_scaled,set_cache=True)
                min_param_bounds = Sclocal.box_scaled[:,0]
                max_param_bounds = Sclocal.box_scaled[:,1]
                pgrad = projection(tr_center_scaled-grad,min_param_bounds,max_param_bounds)-tr_center_scaled
                pgradnorm = np.linalg.norm(pgrad)
                with open(kpstarfile,'r') as f:
                    ds = json.load(f)
                atfidelity = ds["at fidelity"][0]
                if pgradnorm <= min_gradientNorm and atfidelity >= max_fidelity:
                    statusToWrite = 1
                if pgradnorm <= tr_sigma * tr_radius:
                    gradCondToWrite = True
                else:
                    gradCondToWrite = False
                if debug: print(
                    "||pgrad|| \t= %.3f <=> %.3f" % (pgradnorm, tr_sigma * tr_radius))
            except:
                pgradnorm = 1.0
                gradCondToWrite = False
                pass

            sys.stdout.flush()
    gradCondToWrite = comm.bcast(gradCondToWrite, root=0)
    statusToWrite = comm.bcast(statusToWrite, root=0)
    pgradnorm = comm.bcast(pgradnorm, root=0)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                                          value=gradCondToWrite)
    ato.putInMemoryMap(memoryMap=memorymap, key="status",
                       value=statusToWrite)
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientNorm",
                       value=pgradnorm)
    ato.writeMemoryMap(memorymap)
    #comm.barrier() # Maybe redundant. Remove this if testing shows that this is not required
    if debug: print("BYE from approx", rank, gradCondToWrite)
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
    status = ato.getFromMemoryMap(memoryMap=memorymap, key="status")

    if status == 0:
        run_approx(
            memorymap,
            args.EXPDATA,
            args.WEIGHTS
        )

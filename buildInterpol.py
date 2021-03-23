import argparse
import json
import numpy as np
import apprentice.tools as ato

# DO NOT REMOVE COMMENTED CODE FROM THE FUNCTION BELOW
#orc@15-02: looks like only algoparams and newparamoutfile is used as args, hence, omitting the rest
#def buildInterpolationPoints(algoparams,paramfileName,iterationNo,newparamoutfile,prevparamoutfile):
def buildInterpolationPoints(processcard=None,memoryMap=None,newparamoutfile=None,
                             outdir=None,fnamep="params.dat",fnameg="generator.cmd"):
    ############################################################
    # Step 0: Get relevent algorithm parameters and past parameter
    # vectors
    ############################################################
    debug = ato.getFromMemoryMap(memoryMap=memoryMap, key="debug")
    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")
    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    point_min_dist = ato.getFromMemoryMap(memoryMap=memorymap, key="point_min_dist")
    min_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="min_param_bounds")
    max_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="max_param_bounds")



    # prevparamsarr = []
    # fnamearr = []
    # pnoarr = []
    # np_remain = N_p
    # for iter in range(iterationNo):
    #     fname = paramfileName + "_k{}.json".format(iter)
    #     with open(fname,'r') as f:
    #         ds = json.load(f)
    #     pparr = ds['parameters']
    #     for pno, p in enumerate(pparr):
    #         distarr = [np.abs(p[vno] - tr_center[vno]) for vno in range(dim)]
    #         infn = max(distarr)
    #         if infn <= tr_radius:
    #             prevparamsarr.append(p)
    #             fnamearr.append(fname)
    #             pnoarr.append(pno)
    #
    # prevparamobj = {}
    # prevparamsarraccpet = []
    # for pno,p in enumerate(prevparamsarr):
    #     add = True
    #     for pa in prevparamsarraccpet:
    #         distarr = [np.abs(p[vno] - pa[vno]) for vno in range(dim)]
    #         infn = max(distarr)
    #         add = infn >= point_min_dist
    #         if not add:
    #             break
    #     if add:
    #         np_remain -= 1
    #         if fnamearr[pno] not in prevparamobj:
    #             prevparamobj[fnamearr[pno]] = {}
    #         prevparamobj[fnamearr[pno]][str(pnoarr[pno])] = p
    #     if np_remain==0:
    #         break
    # print(prevparamobj)
    # print(np_remain)
    np_remain = N_p
    newparams = None
    minarr = [max(tr_center[d] - tr_radius, min_param_bounds[d]) for d in range(dim)]
    maxarr = [min(tr_center[d] + tr_radius, max_param_bounds[d]) for d in range(dim)]

    if debug: print("TR bounds \t= {}".format([["%.3f"%a,"%.3f"%b] for a,b in zip(minarr,maxarr)]))
    while np_remain >0:
        ############################################################
        # Step 2: get the remaining points needed (doing uniform random
        # for now)
        ############################################################

        Xperdim = ()
        for d in range(dim):
            Xperdim = Xperdim + (np.random.rand(np_remain, ) *
                                 (maxarr[d] - minarr[d]) + minarr[d],)  # Coordinates are generated in [MIN,MAX]

        Xnew = np.column_stack(Xperdim)

        ############################################################
        # Step 3: Make sure all points are at least a certain distance
        # from each other. If not, go to step 2 and repeat
        ############################################################

        for xn in Xnew:
            newparamsAccept = [True]
            newparamsAccept2 = [True]
            # if len(prevparams) > 0:
            #     newparamsAccept = [False] * len(prevparams[prevParamAccept])
            #     for xno,xo in enumerate(prevparams[prevParamAccept]):
            #         distarr = [np.abs(xn[vno] - xo[vno]) for vno in range(dim)]
            #         infn = max(distarr)
            #         newparamsAccept[xno] = infn >= point_min_dist
            if newparams is not None:
                newparamsAccept2 = [False] * len(newparams)
                for xno,xo in enumerate(newparams):
                    distarr = [np.abs(xn[vno] - xo[vno]) for vno in range(dim)]
                    infn = max(distarr)
                    newparamsAccept2[xno] = infn >= point_min_dist

            if all(newparamsAccept) and all(newparamsAccept2):
                if newparams is not None:
                    newparams = np.concatenate((newparams, np.array([xn])))
                else:
                    newparams = np.array([xn])
                np_remain -= 1

            if np_remain == 0:
                break

    ############################################################
    # Step 4: Output all the new points to be given to the problem
    # to run the simulation on
    ############################################################
    ds = {
        "parameters":newparams.tolist()
    }
    # if prevparams is not None:
    #     ds["prevparameters"] = prevparams[prevParamAccept].tolist()
    # print(ds)
    with open(newparamoutfile,'w') as f:
        json.dump(ds, f, indent=4)

    ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                       value=False) #gradCond -> NO
    ato.writeMemoryMap(memorymap)
    #orc@19-03: writePythiaFiles func causing problem w multiple procs
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank ==0:
        ato.writePythiaFiles(processcard,param_names,newparams,outdir,fnamep,fnameg)

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    # parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
    #                     help="Algorithm Parameters (JSON)")
    #parser.add_argument("-p", dest="PARAMFILENAME", type=str, default=None,
    #                    help="Previous parameters file name string before adding the iteration "
    #                         "number and file extention e.g., new_params_N_p") #NOT USED FOR NOW
#    parser.add_argument("--iterno", dest="ITERNO", type=int, default=0,
#                        help="Current iteration number")
#    parser.add_argument("--newpout", dest="NEWPOUTFILE", type=str, default=None,
#                        help="New parameters output file (JSON)")
    #parser.add_argument("--prevpout", dest="PREVPOUTFILE", type=str, default=None,
    #                    help="Previous parameters (to reuse) output file (JSON)") #NOT USED FOR NOW
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")

    args = parser.parse_args()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")

    newparams_Np_k = "logs/newparams_Np" + "_k{}.json".format(k)
    pythiadir_Np_k = "logs/pythia_Np" + "_k{}".format(k)

    buildInterpolationPoints(
        args.PROCESSCARD,
        memorymap,
        newparams_Np_k,
        pythiadir_Np_k
    )


    # buildInterpolationPoints(
    #     args.ALGOPARAMS,
    # #    args.PREVPARAMSFN,
    # #    args.ITERNO,
    #     k,
    #     args.PROCESSCARD,
    #     pythiadir_Np_k,
    #     newparams_Np_k
    # #    args.NEWPOUTFILE
    # #    args.PREVPOUTFILE
    # )

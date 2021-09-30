import argparse
import json
import numpy as np
import apprentice.tools as ato

def checkifnotinminseperationdist(point1,point2,mindist):
    distarr = [np.abs(point1[vno] - point2[vno]) for vno in range(len(point1))]
    infn = max(distarr)
    return infn >= mindist

def readParamFile(file):
    import json
    with open(file, 'r') as f:
        ds = json.load(f)
        return ds['parameters'], ds['at fidelity']

def checkIfPointInTR(point, tr_center, tr_radius):
    distarr = [np.abs(point[vno] - tr_center[vno]) for vno in range(len(point))]
    infn = max(distarr)
    return infn <= tr_radius

def checkIfSamePoint(point1,point2):
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    for vno in range(len(point1)):
        if not isclose(point1[vno],point2[vno]):
            return False
    return True

def addParamsToPool(file,iterno,paramtype,tr_center,tr_radius,p_pool,p_pool_at_fidelity,p_pool_metadata):
    (params,at_fidelity) = readParamFile(file)
    for pno,p in enumerate(params):
        if checkIfPointInTR(p, tr_center, tr_radius):
            if not checkIfSamePoint(p, tr_center):
                if p_pool is None:
                    p_pool = np.array([p])
                    p_pool_metadata["0"]={"file":file,"fidelity to reuse":at_fidelity[pno],"index":pno,
                                            "k":iterno,"ptype":paramtype}
                else:
                    p_pool = np.concatenate((p_pool, np.array([p])))
                    p_pool_metadata[str(len(p_pool)-1)] = {"file": file,
                                                           "fidelity to reuse":at_fidelity[pno],
                                                           "index": pno, "k":iterno,"ptype":paramtype}
                p_pool_at_fidelity.append(at_fidelity[pno])
    return p_pool,p_pool_at_fidelity

def addTrCenterToSelPrev(tr_center,file,iterno,p_sel_prev,p_sel_prev_metadata):
    (params,at_fidelity) = readParamFile(file)
    p_pool_metadata = {"0":{"file":file,"fidelity to reuse":at_fidelity[0],
                            "index":0,"k":iterno,"ptype":"1"}}
    return addParamToSelPrev(tr_center,0,p_pool_metadata,p_sel_prev,p_sel_prev_metadata)

def addParamToSelPrev(param,poolindex,p_pool_metadata,p_sel_prev,p_sel_prev_metadata):
    if p_sel_prev is None:
        p_sel_prev = np.array([param])
        p_sel_prev_metadata['0'] = p_pool_metadata[str(poolindex)]
    else:
        p_sel_prev = np.concatenate((p_sel_prev, np.array([param])))
        p_sel_prev_metadata[str(len(p_sel_prev)-1)] = p_pool_metadata[str(poolindex)]
    return p_sel_prev

def addParamsToSelNew(p_init, I_init, p_sel_prev, p_sel_new,Np,mindist):
    for pino, pi in enumerate(p_init):
        if not I_init[pino]: continue
        result = True
        if p_sel_prev is not None:
            for psprev in p_sel_prev:
                result = checkifnotinminseperationdist(pi, psprev, mindist)
                if not result:
                    break
        if result and p_sel_new is not None:
            for psnew in p_sel_new:
                result = checkifnotinminseperationdist(pi, psnew, mindist)
                if not result:
                    break
        if result:
            if p_sel_new is None:
                p_sel_new = np.array([pi])
            else:
                p_sel_new = np.concatenate((p_sel_new, np.array([pi])))
            I_init[pino] = False
        psnewlength = 0
        psprevlength = 0
        if p_sel_new is not None:
            psnewlength = len(p_sel_new)
        if p_sel_prev is not None:
            psprevlength = len(p_sel_prev)
        if psnewlength + psprevlength == Np:
            break
    return p_sel_new,I_init

def buildInterpolationPoints(processcardarr=None,memoryMap=None,newparamoutfile="newp.json",
                             outdir=None,prevparamoutfile="oldp.json",fnamep="params.dat",
                             fnameg="generator.cmd"):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    comm.barrier()
    ############################################################
    # Get relevent algorithm parameters
    ############################################################
    currIteration = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    debug = True \
        if "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")) \
        else False
    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")
    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    theta = ato.getFromMemoryMap(memoryMap=memorymap, key="theta")
    thetaprime = ato.getFromMemoryMap(memoryMap=memorymap, key="thetaprime")
    min_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="min_param_bounds")
    max_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="max_param_bounds")

    ############################################################
    # Data Structures
    ############################################################
    nptoget = 2 * N_p
    minarr = [max(tr_center[d] - tr_radius, min_param_bounds[d]) for d in range(dim)]
    maxarr = [min(tr_center[d] + tr_radius, max_param_bounds[d]) for d in range(dim)]
    mindist = theta * tr_radius
    equivalencedist = thetaprime * tr_radius


    p_init = np.array([])
    p_pool = None
    p_pool_at_fidelity = []
    p_pool_metadata = {}
    p_sel_prev = None
    p_sel_prev_metadata = {}
    p_sel_new = None
    I_init = []
    I_pool = []


    factor = 1
    while(len(p_init[I_init]) < nptoget):
        ############################################################
        # Inititalize p_init and I_init
        ############################################################
        p_init = ato.getLHSsamples(dim=dim,npoints=factor*nptoget,criterion="maximin",minarr=minarr,maxarr=maxarr)
        I_init = [True for i in p_init]

        ############################################################
        # Discard points from p_init that are in minimum seperation distance
        ############################################################
        for pcurrindex in range(1,len(p_init)):
            pcurr = p_init[pcurrindex]
            for potherindex in range(pcurrindex):
                if not I_init[potherindex]: continue
                pother = p_init[potherindex]
                result = checkifnotinminseperationdist(pcurr,pother,mindist)
                if not result:
                    I_init[pcurrindex] = result
                    break
        factor+=1
    if debug:
        print("P INIT")
        print(p_init)
        print("###############################################")

    ############################################################
    # polulate p_pool
    ############################################################
    tr_center_param_fn = "logs/newparams_1" #+ "_k{}.json".format(k)
    prev_np_param_fn = "logs/newparams_Np" #+ "_k{}.json".format(k)
    for k in range(currIteration):
        tr_center_param = tr_center_param_fn + "_k{}.json".format(k)
        (p_pool,p_pool_at_fidelity) = addParamsToPool(tr_center_param,k,"1",tr_center,tr_radius,
                                                    p_pool,p_pool_at_fidelity,p_pool_metadata)

        prev_np_param =  prev_np_param_fn + "_k{}.json".format(k)
        (p_pool,p_pool_at_fidelity) = addParamsToPool(prev_np_param,k,"Np", tr_center, tr_radius,
                                                    p_pool,p_pool_at_fidelity,p_pool_metadata)

    if p_pool is not None:
        I_pool = [True for i in p_pool]
        ############################################################
        # Discard points from p_pool that are the very similar
        ############################################################
        for pcurrindex in range(1, len(p_pool)):
            pcurr = p_pool[pcurrindex]
            for potherindex in range(pcurrindex):
                if not I_pool[potherindex]: continue
                pother = p_pool[potherindex]
                result = checkIfSamePoint(pcurr, pother)
                if result:
                    I_pool[pcurrindex] = False
                    break

    if debug:
        print("p_pool after initial discard")
        print(p_pool)
        print(p_pool_at_fidelity)
        print(I_pool)
        # print(p_pool_metadata)
        print("###############################################")

    ############################################################
    # Add tr_center to p_sel_prev
    ############################################################
    tr_center_param = tr_center_param_fn + "_k{}.json".format(currIteration)
    p_sel_prev = addTrCenterToSelPrev(tr_center,tr_center_param,currIteration,
                                      p_sel_prev,p_sel_prev_metadata)
    if debug:
        print("p_sel_prev after adding tr_center")
        print(p_sel_prev)
        # print(p_sel_prev_metadata)
        print("###############################################")

    ############################################################
    # Find close matches from p_pool for points in p_init
    # and add to p_sel_prev
    # Sort the p_pool by descending order of p_pool_at_fidelity first
    ############################################################
    if p_pool is not None:
        ppf_order = np.argsort(-1*np.array(p_pool_at_fidelity))[:len(p_pool_at_fidelity)]
        for pino,pi in enumerate(p_init):
            if not I_init[pino]: continue
            for ppno in ppf_order:
                pp = p_pool[ppno]
                if not I_pool[ppno]: continue
                result = checkifnotinminseperationdist(pi, pp, equivalencedist)
                if not result:
                    p_sel_prev = addParamToSelPrev(pp,ppno,p_pool_metadata,p_sel_prev,p_sel_prev_metadata)
                    I_pool[ppno] = False
                    I_init[pino] = False

        if debug:
            print("p_sel_prev after matching close matches")
            print(p_sel_prev)
            # print(p_sel_prev_metadata)
            print(I_pool)
            print(I_init)
            print("###############################################")
    ############################################################
    # If not enough points add points not used before or not in
    # minimum seperation distance from p_pool to p_sel_prev
    # Sort the p_pool by descending order of p_pool_at_fidelity first
    ############################################################
    if p_pool is not None:
        ppf_order = np.argsort(-1*np.array(p_pool_at_fidelity))[:len(p_pool_at_fidelity)]
        if p_sel_prev is None or len(p_sel_prev) < nptoget:
            for ppno in ppf_order:
                pp = p_pool[ppno]
                if not I_pool[ppno]: continue
                result = True
                if p_sel_prev is not None:
                    for psprev in p_sel_prev:
                        result = checkifnotinminseperationdist(pp, psprev, mindist)
                        if not result:
                            break

                if result:
                    p_sel_prev = addParamToSelPrev(pp, ppno, p_pool_metadata, p_sel_prev,
                                                   p_sel_prev_metadata)
                    I_pool[ppno] = False

        if debug:
            print("p_sel_prev after adding points from p_pool")
            print(p_sel_prev)
            # print(p_sel_prev_metadata)
            print(I_pool)
            print("###############################################")

    ############################################################
    # If not enough points add points not used before or not in
    # minimum seperation distance from p_init to p_sel_new
    ############################################################
    if p_sel_prev is None or len(p_sel_prev) < nptoget:
        (p_sel_new, I_init) = addParamsToSelNew(p_init, I_init, p_sel_prev, p_sel_new, nptoget, mindist)

    if debug:
        print("p_sel_new if any are required")
        print(p_sel_new)
        print(I_init)
        print("###############################################")

    ############################################################
    # Save data and exit
    ############################################################
    ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                       value=False)  # gradCond -> NO
    ato.writeMemoryMap(memorymap)

    if rank ==0:
        if p_sel_new is None:
            p_sel_new = np.array([])
        ato.writePythiaFiles(processcardarr, param_names, p_sel_new, outdir, fnamep, fnameg=None)
        ds = {
            "parameters": p_sel_new.tolist(),
            "at fidelity": [0.]*len(p_sel_new)
        }
        with open(newparamoutfile,'w') as f:
            json.dump(ds, f, indent=4)
        if p_sel_prev is None:
            p_sel_prev_metadata["parameters"] = []
        else:
            p_sel_prev_metadata["parameters"] = p_sel_prev.tolist()
        with open(prevparamoutfile,'w') as f:
            json.dump(p_sel_prev_metadata, f, indent=4)


class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-c", dest="PROCESSCARDS", type=str, default=[], nargs='+',
                        help="Process Card location(s) (seperated by a space)")

    args = parser.parse_args()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")

    newparams_Np_k = "logs/newparams_Np" + "_k{}.json".format(k)
    prevparams_Np_k = "logs/prevparams_Np" + "_k{}.json".format(k)
    pythiadir_Np_k = "logs/pythia_Np" + "_k{}".format(k)

    buildInterpolationPoints(
        args.PROCESSCARDS,
        memorymap,
        newparams_Np_k,
        pythiadir_Np_k,
        prevparams_Np_k
    )
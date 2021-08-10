import numpy as np
import json
import argparse
import sys
import apprentice.tools as ato
import apprentice
import glob,os,re
from subprocess import Popen, PIPE
from shutil import copyfile

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

def MCcmd(d,fidelity,loc,MPATH):
    def processreturncode(p,type):
        if p.returncode != 0:
            if debug:print("Running A14 ({}) MC failed with return code {}".format(type,p.returncode))
        return p.returncode

    rivettanalysis = {
        "qcd":"-a ATLAS_2011_S8924791 -a ATLAS_2011_S8971293 -a ATLAS_2011_I919017-a ATLAS_2011_S9128077 -a ATLAS_2012_I1125575 -a ATLAS_2014_I1298811-a ATLAS_2012_I1094564",
        "z":"-a ATLAS_2011_S9131140 -a ATLAS_2014_I1300647",
        "ttbar":"-a ATLAS_2012_I1094568 -a ATLAS_2013_I1243871"
    }

    for r in rivettanalysis.keys():
        runcard = os.path.join(d, "main30_rivet.{}.cmnd".format(r))
        runcardstr = "-p {}".format(runcard)
        fidstr = "-n {}".format(fidelity)
        seedstr = "-s {}".format(str(np.random.randint(1,9999999)))
        outstr = "-o {}.{}".format(loc,r)

        p = Popen(
            [MPATH, runcardstr, fidstr, rivettanalysis[r], seedstr, outstr],
            stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        rc = processreturncode(p,r)
        if rc ==0:
            continue
        else: return rc


    if p.returncode != 0:
        if debug:print("Running miniapp failed with return code {}".format(p.returncode))
    return p.returncode

def mergeyoda(yodafiles,OUTFILE,RBD):
    from subprocess import Popen, PIPE
    if len(yodafiles) == 1:
        copyfile(yodafiles[0], OUTFILE)
    else:
        for filenum,files in enumerate(yodafiles):
            if filenum == len(yodafiles)-1:
                break
            if filenum==0:
                file1=yodafiles[filenum]
                file2=yodafiles[filenum+1]
            else:
                file1=OUTFILE
                file2=yodafiles[filenum+1]
            process = Popen([RBD,'-o',OUTFILE,file1,file2],stdin=PIPE, stdout=PIPE, stderr=PIPE)
            process.communicate()

def selectFilesAndYodaMerge(d,loc,mainfileexists,YPATH):
    yodafiles = []
    mainfile = os.path.join(d, "out_i0.yoda")
    if mainfileexists:
        yodafiles.append(mainfile)
    yodafiles.append(loc)
    outfile = os.path.join(d, "out.yoda")
    mergeyoda(yodafiles,outfile,YPATH)

    os.remove(loc)
    copyfile(outfile,mainfile)
    os.remove(outfile)

def removeYodaDir(rmdirname):
    import os,shutil
    based = os.path.dirname(rmdirname)
    INDIRSLIST = glob.glob(os.path.join(based, "*"))
    dirlist = sorted(INDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    found = False
    for dno,d in enumerate(dirlist):
        if rmdirname == d:
            found = True
        if not found: continue
        else:
            douter = d
            for dnoinner in range(dno+1,len(dirlist)):
                dinner = dirlist[dnoinner]
                shutil.rmtree(douter)
                shutil.copytree(dinner, douter)
                douter = dinner
            shutil.rmtree(douter)
            break

def runMCForAcceptableFidelity(d,atfidelity,bound,fidelity,maxfidelity,pfname,wtfile,
                               usefixedfidelity,MPATH,YPATH,debug):
    def incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,minfidelity,maxfidelity):
        if maxsigma is None or usefixedfidelity: return fidelity
        diff = maxsigma-bound
        newfidelity = int(np.ceil((currfidelity/maxsigma)*diff))
        newfidelity = max(minfidelity,newfidelity)
        if currfidelity+newfidelity > maxfidelity:
            newfidelity = maxfidelity-currfidelity
        return newfidelity

    # size = comm.Get_size()
    # rank = comm.Get_rank()
    maxsigma = None
    # pp=None
    currfidelity = atfidelity
    if currfidelity >= maxfidelity:
        return currfidelity,0
    if rank == 0:
    #     pp = getParameters(d,pfname)
        if currfidelity > 0:
            DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
            sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
            maxsigma = max(sigma)
            sys.stdout.flush()
    maxsigma = comm.bcast(maxsigma, root=0)
    # pp = comm.bcast(pp, root=0)

    rc = 0
    while(maxsigma is None or maxsigma > bound):
        newloc = os.path.join(d, "out_temp.yoda".format(rank))
        newfidelity = None
        if rank == 0:
            newfidelity = incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,100,maxfidelity)
        newfidelity = comm.bcast(newfidelity, root=0)
        rc = MCcmd(d,fidelity=newfidelity,loc=newloc,MPATH=MPATH)
        if not rc == 0:
            break
        comm.barrier()

        if rank == 0:
            currfidelity += newfidelity
            selectFilesAndYodaMerge(d,newloc,mainfileexists=maxsigma is not None,YPATH=YPATH)
            DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
            sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
            maxsigma = max(sigma)
        maxsigma = comm.bcast(maxsigma, root=0)
        currfidelity = comm.bcast(currfidelity, root=0)
        if currfidelity >= maxfidelity or usefixedfidelity:
            break
    return currfidelity,rc

def problem_main_program_parallel_on_Ne(paramfile,prevparamfile,wtfile,memorymap = None,isbebop=False,
                                        outfile=None,outdir=None,pfname="params.dat"):

    rank = comm.Get_rank()
    debug = True \
        if "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")) \
        else False

    if isbebop:
        MPATH = "/home/oyildiz/mohan/pythia/pythia8-diy-master/install/bin/pythia8-diy"
        YPATH = "/home/oyildiz/mohan/mc_miniapp/YODA-1.8.1/bin/yodamerge"
    else:
        if debug: print("Cannot run A14 main locally")
        return 0,0


    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    fidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="fidelity")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")
    usefixedfidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="usefixedfidelity")
    kappa = ato.getFromMemoryMap(memoryMap=memorymap, key="kappa")
    maxfidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="maxfidelity")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    successParams = 0
    totalparams = 0
    indirsAll = None
    atfidelityAll = None
    origfileAll = None
    origfileindexAll = None
    simulationBudgetUsed = 0


    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    min_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="min_param_bounds")
    max_param_bounds = ato.getFromMemoryMap(memoryMap=memorymap,
                                            key="max_param_bounds")

    for d in range(dim):
        if min_param_bounds[d] > tr_center[d] or tr_center[d] > max_param_bounds[d]:
            raise Exception("Starting TR center along dimension {} is not within parameter bound "
                            "[{}, {}]".format(d+1,min_param_bounds[d],max_param_bounds[d]))


    if rank == 0:
        # Following DS only in rank 0
        re_pfname = re.compile(pfname) if pfname else None
        indirsAll = []
        origfileindexAll = []
        origfileAll = []
        atfidelityAll = []
        removedata = {}
        keepdata = {}
        if not usefixedfidelity and prevparamfile is not None:
            with open(prevparamfile,'r') as f:
                prevparamds = json.load(f)
            if len(prevparamds["parameters"]) > 0:
                for pno,param in enumerate(prevparamds["parameters"]):
                    prevk = prevparamds[str(pno)]["k"]
                    prevptype = prevparamds[str(pno)]["ptype"]
                    prevdir = "logs/pythia_{}_k{}".format(prevptype,prevk)
                    INDIRSLIST = glob.glob(os.path.join(prevdir, "*"))
                    dirlist = sorted(INDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
                    previndex = prevparamds[str(pno)]["index"]
                    atfidelityAll.append(prevparamds[str(pno)]["fidelity to reuse"])
                    indirsAll.append(dirlist[previndex])
                    pfm = None
                    files = glob.glob(os.path.join(dirlist[previndex], "*"))
                    for f in files:
                        if re_pfname and re_pfname.search(os.path.basename(f)):
                            pfm = apprentice.io.read_paramsfile(f)
                    if pfm is None:
                        raise Exception("Something went wrong. Cannot get parameter")
                    pp = [pfm[pn] for pn in param_names]
                    if not np.all(np.isclose(param, pp)):
                        raise Exception("Something went wrong. Parameters don't match.\n{}\n{}".format(param,pp))
                    origfileAll.append(prevparamds[str(pno)]["file"])
                    origfileindexAll.append(previndex)
        newINDIRSLIST = glob.glob(os.path.join(outdir, "*"))
        dirlist = sorted(newINDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        for dno,d in enumerate(dirlist):
            indirsAll.append(d)
            origfileindexAll.append(dno)
            origfileAll.append(paramfile)
            atfidelityAll.append(0)
        totalparams = len(atfidelityAll)
    totalparams = comm.bcast(totalparams, root=0)
    indirsAll = comm.bcast(indirsAll, root=0)
    atfidelityAll = comm.bcast(atfidelityAll, root=0)
    origfileAll = comm.bcast(origfileAll, root=0)
    origfileindexAll = comm.bcast(origfileindexAll, root=0)

    runatfidelity = None
    runatfidelityFound = False
    for currParamIndex in range(totalparams):
        if successParams >= N_p:
            break
        d = indirsAll[currParamIndex]
        atfid = atfidelityAll[currParamIndex]
        of = origfileAll[currParamIndex]
        ofi = origfileindexAll[currParamIndex]
        # TODO reuse fidelity for 2nd parameter onward
        # if not runatfidelityFound:
        (cfd,rc) = runMCForAcceptableFidelity(d,atfidelity=atfid,bound=kappa*(tr_radius**2),fidelity=fidelity,
                                              maxfidelity=maxfidelity,pfname=pfname,wtfile=wtfile,
                                              usefixedfidelity=usefixedfidelity, MPATH=MPATH,debug=debug)
            # if rc == 0:
            #     runatfidelity = cfd
            #     runatfidelityFound = True
        # else:
        #     (cfd,rc) = runMCAtFidelity(d,atfidelity=atfid,runatfidelity=runatfidelity,
        #                                         pfname=pfname,MPATH=MPATH,debug=debug)
        if rank == 0:
            if rc == 0:
                with open(of,'r') as f:
                    ds = json.load(f)
                ds["at fidelity"][ofi] = cfd
                if of in keepdata:
                    keepdata[of]["ofi"].append(ofi)
                    keepdata[of]["d"].append(d)
                else:
                    keepdata[of] = {"ofi":[ofi],"d":[d]}
                with open(of,'w') as f:
                    json.dump(ds,f,indent=4)
            else:
                if of in removedata:
                    removedata[of]["ofi"].append(ofi)
                    removedata[of]["d"].append(d)
                else:
                    removedata[of] = {"ofi":[ofi],"d":[d]}

        if rc==0:
            simulationBudgetUsed = simulationBudgetUsed + cfd - atfid
            successParams += 1
    if rank == 0:
        with open(paramfile,'r') as f:
            newds = json.load(f)
        if len(newds["parameters"]) >0:
            startindex = 0
            if paramfile in keepdata:
                arr = [int(i) for i in keepdata[paramfile]["ofi"]]
                startindex = max(arr)+1
            for i in range(startindex,len(newds["parameters"])):
                for d,of,ofi in zip(indirsAll,origfileAll,origfileindexAll):
                    if of == paramfile and ofi == i:
                        if paramfile in removedata:
                            if i not in removedata[paramfile]["ofi"] and d not in removedata[paramfile]["d"]:
                                removedata[paramfile]["ofi"].append(i)
                                removedata[paramfile]["d"].append(d)
                        else:
                            removedata[paramfile] = {"ofi":[i],"d":[d]}

        for pf in removedata.keys():
            with open (pf,'r') as f:
                ds = json.load(f)
            for ofi,d in zip(reversed(removedata[pf]["ofi"]),reversed(removedata[pf]["d"])):
                del ds["parameters"][ofi]
                del ds["at fidelity"][ofi]
                removeYodaDir(d)
            with open(pf,'w') as f:
                json.dump(ds,f,indent=4)
    if debug:
        print("mc_miniapp done.")
        sys.stdout.flush()

    return simulationBudgetUsed,successParams



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Simulation',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-o", dest="OPTION", type=str, default=None,
                        help="Option (initial,single, or multi)")
    parser.add_argument("-c", dest="PROCESSCARDS", type=str, default=[], nargs='+',
                        help="Process Card location(s) (seperated by a space)")
    parser.add_argument("-b", "--bebop", dest="BEBOP", default=False, action="store_true",
                        help="Running on BEBOP")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default="conf/weights",
                        help="Weights file (TXT)")

    args = parser.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.barrier()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    debug = True \
        if "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")) \
        else False
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
        prevparamfile = None
        paramfile = "logs/newparams_1_k0.json"
        outfile = "logs/MCout_1_k0.h5"
        outdir  = "logs/pythia_1_k0"

        simulationBudgetUsed = 0
        successParams = 0
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
                    json.dump({"parameters": [tr_center],"at fidelity":[0.]}, f, indent=4)
                ato.writePythiaFiles(args.PROCESSCARDS, param_names, [tr_center],
                                     outdir)

            (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Ne(
                paramfile,
                prevparamfile,
                args.WEIGHTS,
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
            statusToWrite = 4 if successParams < 1 else 0
            ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                               value=simulationBudgetUsed)
            ato.putInMemoryMap(memoryMap=memorymap, key="status",
                               value=statusToWrite)
            ato.writeMemoryMap(memoryMap=memorymap)

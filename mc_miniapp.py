import shutil

import numpy as np
import json
import argparse
import sys
import apprentice.tools as ato
import apprentice
import glob,os,re
from subprocess import Popen, PIPE
from shutil import copyfile

# Keep at outermost level
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

# Keep at outermost level
def getParameters(d, pfname):
    re_pfname = re.compile(pfname) if pfname else None
    files = glob.glob(os.path.join(d, "*"))
    param = None
    for f in files:
        if re_pfname and re_pfname.search(os.path.basename(f)):
            param = apprentice.io.read_paramsfile(f)
    # if param is None:
    #     if debug:print("Something went wrong. Cannot get parameter")
    #     return currfidelity,-1
    pp = [param[pn] for pn in param_names]
    return pp

# Keep at outermost level
def MCcmd(pp,fidelity,loc,MPATH):
    p = Popen(
        [MPATH, str(pp[0]), str(pp[1]), str(pp[2]),
         str(fidelity), str(np.random.randint(1,9999999)), "0", "1", loc],
        stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.communicate(b"input data that is passed to subprocess' stdin")
    if p.returncode != 0:
        if debug:print("Running miniapp failed with return code {}".format(p.returncode))
    return p.returncode

# Keep at outermost level
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

# Keep at outermost level
def problem_main_program_parallel_on_Ne(paramfile,prevparamfile,wtfile,memorymap = None,isbebop=False,
                         outfile=None,outdir=None,pfname="params.dat"):
    # Keep at parallel on Ne level (main fn)
    def chunkfidelity(newfidelity,minfidelity):
        size = comm.Get_size()
        splitfidelity = np.ceil(newfidelity/size)
        if splitfidelity >minfidelity:
            newfidelityArr = [int(splitfidelity)] * size
        else:
            newfidelityArr = [0] * size
            fidremain = newfidelity
            for rank in range(size):
                if fidremain < minfidelity:
                    newfidelityArr[rank] = minfidelity
                    break
                newfidelityArr[rank] = minfidelity
                fidremain -= minfidelity
        return newfidelityArr

    # Keep at parallel on Ne level (main fn)
    def selectFilesAndYodaMerge(d,newfidelityArr,mainfileexists,YPATH):
        yodafiles = []
        mainfile = os.path.join(d, "out_i0.yoda")
        if mainfileexists:
            yodafiles.append(mainfile)
        for idx,f in enumerate(newfidelityArr):
            if f != 0:
                newloc = os.path.join(d, "out_temp_r{}.yoda".format(idx))
                yodafiles.append(newloc)
        outfile = os.path.join(d, "out.yoda")
        mergeyoda(yodafiles,outfile,YPATH)
        for idx,f in enumerate(newfidelityArr):
            if f != 0:
                newloc = os.path.join(d, "out_temp_r{}.yoda".format(idx))
                os.remove(newloc)
        copyfile(outfile,mainfile)
        os.remove(outfile)

    # Keep at parallel on Ne level (main fn)
    def runMCForAcceptableFidelity(d,atfidelity,bound,fidelity,maxfidelity,pfname,wtfile,
                                   usefixedfidelity,MPATH,YPATH,debug):
        def incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,minfidelity,maxfidelity):
            if maxsigma is None or usefixedfidelity: return chunkfidelity(fidelity,minfidelity)
            diff = maxsigma-bound
            newfidelity = int(np.ceil((currfidelity/maxsigma)*diff))
            newfidelity = max(minfidelity,newfidelity)
            if currfidelity+newfidelity > maxfidelity:
                newfidelity = maxfidelity-currfidelity
            return chunkfidelity(newfidelity,minfidelity)

        size = comm.Get_size()
        rank = comm.Get_rank()
        maxsigma = None
        pp=None
        currfidelity = atfidelity
        if currfidelity >= maxfidelity:
            return currfidelity,np.zeros(size)
        if rank == 0:
            pp = getParameters(d,pfname)
            if currfidelity > 0:
                DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
                sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
                maxsigma = max(sigma)
                sys.stdout.flush()
        maxsigma = comm.bcast(maxsigma, root=0)
        pp = comm.bcast(pp, root=0)

        returncodes = np.zeros(size)
        while(maxsigma is None or maxsigma > bound):
            newloc = os.path.join(d, "out_temp_r{}.yoda".format(rank))
            newfidelityArr = None
            if rank == 0:
                newfidelityArr = incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,100,maxfidelity)
            newfidelityArr = comm.bcast(newfidelityArr, root=0)
            if newfidelityArr[rank] !=0:
                returncodes[rank] = MCcmd(pp,fidelity=newfidelityArr[rank],loc=newloc,MPATH=MPATH)
            if not np.all((returncodes == 0)):
                break
            comm.barrier()

            if rank == 0:
                currfidelity += sum(newfidelityArr)
                selectFilesAndYodaMerge(d,newfidelityArr,mainfileexists=maxsigma is not None,YPATH=YPATH)
                DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
                sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
                maxsigma = max(sigma)
            maxsigma = comm.bcast(maxsigma, root=0)
            currfidelity = comm.bcast(currfidelity, root=0)
            if currfidelity >= maxfidelity or usefixedfidelity:
                break
        return currfidelity,returncodes

    # Keep at parallel on Ne level (main fn)
    def runMCAtFidelity(d,atfidelity,runatfidelity,pfname,MPATH,YPATH,debug):
        size = comm.Get_size()
        rank = comm.Get_rank()
        pp=None
        currfidelity = atfidelity
        if currfidelity >= runatfidelity:
            return currfidelity,np.zeros(size)
        if rank == 0: pp = getParameters(d,pfname)
        pp = comm.bcast(pp, root=0)
        returncodes = np.zeros(size)
        newloc = os.path.join(d, "out_temp_r{}.yoda".format(rank))
        newfidelityArr = None
        if rank == 0:
            newfidelityArr = chunkfidelity(runatfidelity,100)
        newfidelityArr = comm.bcast(newfidelityArr, root=0)
        if newfidelityArr[rank] !=0: returncodes[rank] = MCcmd(pp,fidelity=newfidelityArr[rank],loc=newloc,MPATH=MPATH)
        comm.barrier()

        if rank == 0:
            selectFilesAndYodaMerge(d,newfidelityArr,mainfileexists=atfidelity>0,YPATH=YPATH)
            currfidelity += sum(newfidelityArr)
        currfidelity = comm.bcast(currfidelity, root=0)
        return currfidelity,returncodes

    rank = comm.Get_rank()

    if isbebop:
        MPATH = "/home/oyildiz/mohan/mc_miniapp/pythia8rivetminiapp/miniapp"
        YPATH = "/home/oyildiz/mohan/mc_miniapp/YODA-1.8.1/bin/yodamerge"
    else:
        MPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/pythia8rivetminiapp/miniapp"
        YPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/YODA-1.8.1/bin/yodamerge"

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

    debug = True \
        if "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")) \
        else False

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
        if not runatfidelityFound:
            (cfd,returncodes) = runMCForAcceptableFidelity(d,atfidelity=atfid,bound=kappa*(tr_radius**2),fidelity=fidelity,
                                                  maxfidelity=maxfidelity,pfname=pfname,wtfile=wtfile,
                                                  usefixedfidelity=usefixedfidelity, MPATH=MPATH,YPATH=YPATH,debug=debug)
            if np.all(returncodes==0):
                runatfidelity = cfd
                runatfidelityFound = True
        else:
            (cfd,returncodes) = runMCAtFidelity(d,atfidelity=atfid,runatfidelity=runatfidelity,
                                                pfname=pfname,MPATH=MPATH,YPATH=YPATH,debug=debug)
        if rank == 0:
            if np.all(returncodes==0):
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

        if np.all(returncodes==0):
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

# Keep at outermost level
def problem_main_program_parallel_on_Np(paramfile,prevparamfile,wtfile,memorymap = None,isbebop=False,
                         outfile=None,outdir=None,pfname="params.dat"):
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
    def runMCForAcceptableFidelity(d,atfidelity,bound,fidelity,maxfidelity,pfname,wtfile,
                                   usefixedfidelity,MPATH,YPATH,debug):
        def incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,minfidelity,maxfidelity):
            if maxsigma is None or usefixedfidelity: return fidelity
            diff = maxsigma-bound
            newfidelity = int(np.ceil((currfidelity/maxsigma)*diff))
            newfidelity = max(minfidelity,newfidelity)
            if currfidelity+newfidelity > maxfidelity:
                return maxfidelity-currfidelity
            else:
                return newfidelity

        currfidelity = atfidelity
        if currfidelity >= maxfidelity:
            return currfidelity,0
        pp = getParameters(d,pfname)
        maxsigma = None
        if currfidelity > 0:
            DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
            sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
            maxsigma = max(sigma)
            sys.stdout.flush()
        while(maxsigma is None or maxsigma > bound):
            newfidelity = incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,50,maxfidelity)
            newloc = os.path.join(d, "out_temp.yoda")
            # p = Popen(
            #     [MPATH, str(pp[0]), str(pp[1]), str(pp[2]),
            #      str(newfidelity), str(np.random.randint(1,9999999)), "0", "1", newloc],
            #     stdin=PIPE, stdout=PIPE, stderr=PIPE)
            # p.communicate(b"input data that is passed to subprocess' stdin")
            rc = MCcmd(pp,fidelity=newfidelity,loc=newloc,MPATH=MPATH)
            if rc != 0:
                return currfidelity,rc
            currfidelity += newfidelity
            selectFilesAndYodaMerge(d,newloc,maxsigma is not None,YPATH)
            DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
            sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
            maxsigma = max(sigma)

            if currfidelity >= maxfidelity or usefixedfidelity:
                break
        return currfidelity,0

    def runMCAtFidelity(d,atfidelity,runatfidelity,pfname,MPATH,YPATH,debug):
        currfidelity = atfidelity
        if currfidelity >= runatfidelity:
            return currfidelity,0
        pp = getParameters(d,pfname)
        newloc = os.path.join(d, "out_temp.yoda")
        rc = MCcmd(pp,fidelity=runatfidelity,loc=newloc,MPATH=MPATH)
        if rc != 0:
            return currfidelity,rc
        currfidelity += runatfidelity
        selectFilesAndYodaMerge(d,newloc,currfidelity>0,YPATH)
        return currfidelity,0

    size = comm.Get_size()
    rank = comm.Get_rank()

    if isbebop:
        MPATH = "/home/oyildiz/mohan/mc_miniapp/pythia8rivetminiapp/miniapp"
        YPATH = "/home/oyildiz/mohan/mc_miniapp/YODA-1.8.1/bin/yodamerge"
    else:
        MPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/pythia8rivetminiapp/miniapp"
        YPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/YODA-1.8.1/bin/yodamerge"

    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")
    fidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="fidelity")
    dim = ato.getFromMemoryMap(memoryMap=memorymap, key="dim")
    usefixedfidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="usefixedfidelity")
    kappa = ato.getFromMemoryMap(memoryMap=memorymap, key="kappa")
    maxfidelity = ato.getFromMemoryMap(memoryMap=memorymap, key="maxfidelity")
    N_p = ato.getFromMemoryMap(memoryMap=memorymap, key="N_p")
    successParams = 0
    currParamIndex = 0
    totalparams = 0

    debug = True \
        if "All" in ato.getOutlevelDef(ato.getFromMemoryMap(memoryMap=memorymap, key="outputlevel")) \
        else False

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

    simulationBudgetUsed = 0
    while successParams < N_p:
        if currParamIndex == totalparams:
            break
        indirs = None
        origfileindex = None
        origfile = None
        atfidelity = None
        if rank == 0:
            indirs = []
            origfileindex = []
            origfile = []
            atfidelity = []
            for i in range(currParamIndex,min(totalparams,currParamIndex+N_p-successParams)):
                indirs.append(indirsAll[i])
                origfileindex.append(origfileindexAll[i])
                origfile.append(origfileAll[i])
                atfidelity.append(atfidelityAll[i])
            currParamIndex = min(totalparams,currParamIndex+N_p-successParams)

        currParamIndex = comm.bcast(currParamIndex, root=0)
        indirs = comm.bcast(indirs, root=0)
        origfileindex = comm.bcast(origfileindex, root=0)
        origfile = comm.bcast(origfile, root=0)
        atfidelity = comm.bcast(atfidelity, root=0)

        rankDirs = ato.chunkIt(indirs, size) if rank == 0 else None
        rankorigfileindex = ato.chunkIt(origfileindex, size) if rank == 0 else None
        rankorigfile = ato.chunkIt(origfile, size) if rank == 0 else None
        rankatfidelity = ato.chunkIt(atfidelity, size) if rank == 0 else None

        rankDirs = comm.scatter(rankDirs, root=0)
        rankorigfileindex = comm.scatter(rankorigfileindex, root=0)
        rankorigfile = comm.scatter(rankorigfile, root=0)
        rankatfidelity = comm.scatter(rankatfidelity, root=0)

        currfidelity = {}
        runatfidelity = None
        runatfidelityFound = False
        for num, (d,ofi,of,atfid) in enumerate(zip(rankDirs,rankorigfileindex,rankorigfile,rankatfidelity)):
            if not runatfidelityFound:
                (cfd,rc) = runMCForAcceptableFidelity(d,atfidelity=atfid,bound=kappa*(tr_radius**2),fidelity=fidelity,
                                                      maxfidelity=maxfidelity,pfname=pfname,wtfile=wtfile,
                                                      usefixedfidelity=usefixedfidelity, MPATH=MPATH,YPATH=YPATH,debug=debug)
                if rc == 0:
                    runatfidelity = cfd
                    runatfidelityFound = True
            else:
                (cfd,rc) = runMCAtFidelity(d,atfidelity=atfid,runatfidelity=runatfidelity,pfname=pfname,
                                                      MPATH=MPATH,YPATH=YPATH,debug=debug)
            currfidelity["{}_{}".format(of,ofi)] = {"cfd":cfd,"rc":rc}
        currfidelityr0 = comm.gather(currfidelity, root=0)

        if rank==0:
            allcurrfidelity = {}
            for cf in currfidelityr0:
                allcurrfidelity.update(cf)

            of_done = []
            for num, (d,ofi,of,atfid) in enumerate(zip(indirs,origfileindex,origfile,atfidelity)):
                if of in of_done:
                    continue
                with open(of,'r') as f:
                    ds = json.load(f)
                if allcurrfidelity["{}_{}".format(of,ofi)]["rc"] == 0:
                    ds["at fidelity"][ofi] = allcurrfidelity["{}_{}".format(of,ofi)]["cfd"]
                    simulationBudgetUsed += allcurrfidelity["{}_{}".format(of,ofi)]["cfd"] - atfid
                    successParams += 1
                    if of in keepdata:
                        keepdata[of]["ofi"].append(ofi)
                        keepdata[of]["d"].append(d)
                    else:
                        keepdata[of] = {"ofi":[ofi],"d":[d]}
                else:
                    if of in removedata:
                        removedata[of]["ofi"].append(ofi)
                        removedata[of]["d"].append(d)
                    else:
                        removedata[of] = {"ofi":[ofi],"d":[d]}

                for numinner in range(num+1,len(origfile)):
                    ofinner = origfile[numinner]
                    if ofinner in of_done:
                        continue
                    if ofinner == of:
                        ofiinner = origfileindex[numinner]
                        atfidinner = atfidelity[numinner]
                        dinner = indirs[numinner]
                        if allcurrfidelity["{}_{}".format(ofinner,ofiinner)]["rc"] == 0:
                            ds["at fidelity"][ofiinner] = allcurrfidelity["{}_{}".format(ofinner,ofiinner)]["cfd"]
                            simulationBudgetUsed += allcurrfidelity["{}_{}".format(ofinner,ofiinner)]["cfd"] - atfidinner
                            successParams += 1
                            if ofinner in keepdata:
                                keepdata[ofinner]["ofi"].append(ofiinner)
                                keepdata[ofinner]["d"].append(dinner)
                            else:
                                keepdata[ofinner] = {"ofi":[ofiinner],"d":[dinner]}
                        else:
                            if ofinner in removedata:
                                removedata[ofinner]["ofi"].append(ofiinner)
                                removedata[ofinner]["d"].append(dinner)
                            else:
                                removedata[ofinner] = {"ofi":[ofiinner],"d":[dinner]}
                with open(of,'w') as f:
                    json.dump(ds,f,indent=4)
                of_done.append(of)
                if successParams >= N_p:
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
                                        removedata[paramfile]["ofi"].append(i)
                                        removedata[paramfile]["d"].append(d)
                                    else:
                                        removedata[paramfile] = {"ofi":[i],"d":[d]}
        successParams = comm.bcast(successParams, root=0)
        simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)

    if rank == 0:
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

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

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
            simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
            ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                               value=simulationBudgetUsed)
            ato.putInMemoryMap(memoryMap=memorymap, key="status",
                               value=statusToWrite)
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
        status = ato.getFromMemoryMap(memoryMap=memorymap, key="status")

        if args.OPTION == "multi":
            prevparamfile = "logs/prevparams_Np" + "_k{}.json".format(k)
            paramfile = newparams_Np_k
            outfile = MCout_Np_k
            outdir = outdir_Np_k
            if not gradCond and status == 0:
                (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Np(
                    paramfile,
                    prevparamfile,
                    args.WEIGHTS,
                    memorymap,
                    args.BEBOP,
                    outfile,
                    outdir
                )
                statusToWrite = 4 if successParams < N_p else 0
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.putInMemoryMap(memoryMap=memorymap, key="status",
                                   value=statusToWrite)
                ato.writeMemoryMap(memoryMap=memorymap)
        else:
            prevparamfile = None
            paramfile = newparams_1_kp1
            outfile = MCout_1_kp1
            outdir = outdir_1_kp1
            simulationBudgetUsed = 0
            successParams = 0
            if not gradCond and status == 0:
                if rank >= 0:
                    (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Ne(
                        paramfile,
                        prevparamfile,
                        args.WEIGHTS,
                        memorymap,
                        args.BEBOP,
                        outfile,
                        outdir
                    )
                statusToWrite = 4 if successParams < 1 else 0
                simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.putInMemoryMap(memoryMap=memorymap, key="status",
                                   value=statusToWrite)
                ato.writeMemoryMap(memoryMap=memorymap)






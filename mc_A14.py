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

def MCcmd(d,fidelity,loc,numprocs,MPATH):
    def processreturncode(p,type):
        if p.returncode != 0:
            if debug:print("Running A14 ({}) MC failed with return code {}".format(type,p.returncode))
        return p.returncode

    rivettanalysis = {
        "qcd":["ATLAS_2011_S8924791", "ATLAS_2011_S8971293","ATLAS_2011_I919017","ATLAS_2011_S9128077","ATLAS_2012_I1125575","ATLAS_2014_I1298811","ATLAS_2012_I1094564"],
        # "qcd":["ATLAS_2011_S8924791"], #shortened
        "z":["ATLAS_2011_S9131140","ATLAS_2014_I1300647"],
        # "z":["ATLAS_2011_S9131140"], #shortened
        "ttbar":["ATLAS_2012_I1094568","ATLAS_2013_I1243871"]
    }

    if debug: print("Running directory: ",d,end="\r".format(d))
    for r in rivettanalysis.keys():
        runcard = os.path.join(d, "main30_rivet.{}.cmnd".format(r))
        runcardstr = "{}".format(runcard)
        fidstr = "{}".format(fidelity)
        seedstr = "{}".format(str(np.random.randint(1,9999999)))
        outstr = "{}.{}".format(loc,r)

        argarr = ["mpirun","-n",str(numprocs),MPATH, "-p",runcardstr, "-n",fidstr, "-s",seedstr, "-o",outstr]
        for ra in rivettanalysis[r]:
            argarr.append("-a")
            argarr.append(ra)
        # print(argarr)
        # sys.stdout.flush()
        #### UNCOMMENT 1 START ###
    #     import subprocess
    #     p = subprocess.run(argarr,stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    #     print("##########")
    #     print(p.stdout)
    #     print(p.returncode)
    #     print("##########")
    # exit(1)
        #### UNCOMMENT 1 END ###

        #### OR UNCOMMENT 2 START ###
        p = Popen(argarr,stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        rc = processreturncode(p,r)

        if rc == 0:
            continue
        else:
            if debug:
                print("rc was non zero ({}) for argument array:".format(rc))
                print(argarr)
            return rc
        #### UNCOMMENT 2 END ###
    with open(loc, 'w') as outfile:
        for rno,r in enumerate(rivettanalysis.keys()):
            fname = "{}.{}".format(loc,r)
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write("\n")
            # os.remove(fname) #UNCOKMENT ME
    # exit(1)
    # print(d)
    # YPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/YODA-1.8.1/bin/yodamerge"
    # # mergeyoda([os.path.join(d,"out_temp_qcd.yoda"),os.path.join(d,"out_temp_qcd1.yoda")],os.path.join(d,"yodamerge.yoda"),YPATH)
    # mergeyoda([loc,loc],os.path.join(d,"yodamerge.yoda"),YPATH)
    # exit(1)
    return 0

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
                               usefixedfidelity,numprocs,MPATH,YPATH,debug):
    def incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,minfidelity,maxfidelity):
        if maxsigma is None or usefixedfidelity: return fidelity
        diff = maxsigma-bound
        newfidelity = int(np.ceil((currfidelity/maxsigma)*diff))
        newfidelity = max(minfidelity,newfidelity)
        if currfidelity+newfidelity > maxfidelity:
            newfidelity = maxfidelity-currfidelity
        return newfidelity

    maxsigma = None
    currfidelity = atfidelity
    if currfidelity >= maxfidelity:
        return currfidelity,0
    if currfidelity > 0:
        (DATA,BNAMES) = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
        sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
        maxsigma = max(sigma)
        sys.stdout.flush()

    rc = 0
    while(maxsigma is None or maxsigma > bound):
        newloc = os.path.join(d, "out_temp.yoda")
        newfidelity = incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,100,maxfidelity)
        rc = MCcmd(d,fidelity=newfidelity,loc=newloc,numprocs=numprocs,MPATH=MPATH)
        if not rc == 0:
            break

        currfidelity += newfidelity
        selectFilesAndYodaMerge(d,newloc,mainfileexists=maxsigma is not None,YPATH=YPATH)
        (DATA,BNAMES) = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
        sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
        maxsigma = max(sigma)
        if currfidelity >= maxfidelity or usefixedfidelity:
            break
    return currfidelity,rc

def runMCAtFidelity(d,atfidelity,runatfidelity,pfname,numprocs,MPATH,YPATH,debug):
    currfidelity = atfidelity
    if currfidelity >= runatfidelity:
        return currfidelity,0
    newloc = os.path.join(d, "out_temp.yoda")
    rc = MCcmd(d,fidelity=runatfidelity,loc=newloc,numprocs=numprocs,MPATH=MPATH)
    if rc != 0:
        return currfidelity,rc
    selectFilesAndYodaMerge(d,newloc,mainfileexists=atfidelity>0,YPATH=YPATH)
    currfidelity += runatfidelity
    return currfidelity,rc

def problem_main_program_parallel_on_Ne(paramfile,prevparamfile,wtfile,memorymap = None,isbebop=False,
                                        numprocs=36, outfile=None,outdir=None,pfname="params.dat"):

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
            (cfd,rc) = runMCForAcceptableFidelity(d,atfidelity=atfid,bound=kappa*(tr_radius**2),fidelity=fidelity,
                                              maxfidelity=maxfidelity,pfname=pfname,wtfile=wtfile,
                                              usefixedfidelity=usefixedfidelity, numprocs=numprocs,
                                              MPATH=MPATH, YPATH=YPATH,debug=debug)
            if rc == 0:
                runatfidelity = cfd
                runatfidelityFound = True
        else:
            (cfd,rc) = runMCAtFidelity(d,atfidelity=atfid,runatfidelity=runatfidelity,pfname=pfname,
                                       numprocs=numprocs,MPATH=MPATH,YPATH=YPATH,debug=debug)
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
    parser.add_argument("-n", dest="NUMPROCS", type=int, default=36,
                        help="Number of MPI ranks")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default="conf/weights",
                        help="Weights file (TXT)")

    args = parser.parse_args()

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
            if debug:
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

            with open(paramfile, 'w') as f:
                json.dump({"parameters": [tr_center],"at fidelity":[0.]}, f, indent=4)
            ato.writePythiaFiles(args.PROCESSCARDS, param_names, [tr_center],
                                 outdir, fnameg=None)

            (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Ne(
                paramfile,
                prevparamfile,
                args.WEIGHTS,
                memorymap,
                args.BEBOP,
                args.NUMPROCS,
                outfile,
                outdir
            )
            statusToWrite = 4 if successParams < 1 else 0
            ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                               value=simulationBudgetUsed)
            ato.putInMemoryMap(memoryMap=memorymap, key="status",
                               value=statusToWrite)
            ato.writeMemoryMap(memoryMap=memorymap)
        else:
            if debug:
                print("Skipping the initial MC run since k (neq 0) = {}".format(k))
                sys.stdout.flush()
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
                (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Ne(
                    paramfile,
                    prevparamfile,
                    args.WEIGHTS,
                    memorymap,
                    args.BEBOP,
                    args.NUMPROCS,
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
                (simulationBudgetUsed,successParams) = problem_main_program_parallel_on_Ne(
                    paramfile,
                    prevparamfile,
                    args.WEIGHTS,
                    memorymap,
                    args.BEBOP,
                    args.NUMPROCS,
                    outfile,
                    outdir
                )
                statusToWrite = 4 if successParams < 1 else 0
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.putInMemoryMap(memoryMap=memorymap, key="status",
                                   value=statusToWrite)
                ato.writeMemoryMap(memoryMap=memorymap)


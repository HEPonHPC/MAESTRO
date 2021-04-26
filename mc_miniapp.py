import numpy as np
import json
import argparse
import sys
import apprentice.tools as ato
import apprentice
import glob,os,re
from subprocess import Popen, PIPE
from shutil import copyfile

def incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,minfidelity,maxfidelity):
    if maxsigma is None or usefixedfidelity: return fidelity
    diff = maxsigma-bound
    newfidelity = int((currfidelity/maxsigma)*diff)
    newfidelity = max(minfidelity,newfidelity)
    if currfidelity+newfidelity > maxfidelity:
        return maxfidelity-currfidelity
    else:
        return newfidelity

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

def runMCForAcceptableFidelity(d,atfidelity,bound,fidelity,maxfidelity,pfname,wtfile,
                               usefixedfidelity,MPATH,YPATH):
    re_pfname = re.compile(pfname) if pfname else None
    currfidelity = atfidelity
    sys.stdout.flush()
    files = glob.glob(os.path.join(d, "*"))
    param = None
    for f in files:
        if re_pfname and re_pfname.search(os.path.basename(f)):
            param = apprentice.io.read_paramsfile(f)
    if param is None:
        raise Exception("Something went wrong. Cannot get parameter")
    pp = [param[pn] for pn in param_names]
    maxsigma = None
    if currfidelity > 0:
        DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
        sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
        maxsigma = max(sigma)
        sys.stdout.flush()
    while(maxsigma is None or maxsigma > bound):
        newfidelity = incrementfidelity(maxsigma,bound,usefixedfidelity,currfidelity,fidelity,50,maxfidelity)
        newloc = os.path.join(d, "out_temp.yoda")
        p = Popen(
            [MPATH, str(pp[0]), str(pp[1]), str(pp[2]),
             str(newfidelity), str(np.random.randint(1,9999999)), "0", "1", newloc],
            stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        if p.returncode != 0:
            raise Exception("Running miniapp failed with return code {}".format(p.returncode))
        yodafiles = []
        mainfile = os.path.join(d, "out_i0.yoda")
        if maxsigma is not None:
            yodafiles.append(mainfile)
        yodafiles.append(newloc)
        outfile = os.path.join(d, "out.yoda")
        mergeyoda(yodafiles,outfile,YPATH)
        currfidelity += newfidelity
        os.remove(newloc)
        copyfile(outfile,mainfile)
        os.remove(outfile)
        DATA = apprentice.io.readSingleYODAFile(d, pfname, wtfile)
        sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
        maxsigma = max(sigma)

        if currfidelity >= maxfidelity:
            break
    return currfidelity

def problem_main_program(paramfile,prevparamfile,wtfile,memorymap = None,isbebop=False,
                         outfile=None,outdir=None,pfname="params.dat"):
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

    indirs = None
    origfileindex = None
    origfile = None
    atfidelity = None
    re_pfname = re.compile(pfname) if pfname else None
    if rank == 0:
        indirs = []
        origfileindex = []
        origfile = []
        atfidelity = []
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
                    atfidelity.append(prevparamds[str(pno)]["fidelity to reuse"])
                    indirs.append(dirlist[previndex])
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
                    origfile.append(prevparamds[str(pno)]["file"])
                    origfileindex.append(previndex)
        newINDIRSLIST = glob.glob(os.path.join(outdir, "*"))
        dirlist = sorted(newINDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        for dno,d in enumerate(dirlist):
            indirs.append(d)
            origfileindex.append(dno)
            origfile.append(paramfile)
            atfidelity.append(0)

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
    for num, (d,ofi,of,atfid) in enumerate(zip(rankDirs,rankorigfileindex,rankorigfile,rankatfidelity)):
        cfd = runMCForAcceptableFidelity(d,atfidelity=atfid,bound=kappa*(tr_radius**2),fidelity=fidelity,
                                    maxfidelity=maxfidelity,pfname=pfname,wtfile=wtfile,
                                    usefixedfidelity=usefixedfidelity, MPATH=MPATH,YPATH=YPATH)
        currfidelity["{}_{}".format(of,ofi)] = cfd
    currfidelityr0 = comm.gather(currfidelity, root=0)
    simulationBudgetUsed = None
    if rank==0:
        allcurrfidelity = {}
        for cf in currfidelityr0:
            allcurrfidelity.update(cf)

        simulationBudgetUsed = 0
        of_done = []
        for num, (d,ofi,of,atfid) in enumerate(zip(indirs,origfileindex,origfile,atfidelity)):
            if of in of_done:
                continue
            with open(of,'r') as f:
                ds = json.load(f)
            ds["at fidelity"][ofi] = allcurrfidelity["{}_{}".format(of,ofi)]
            simulationBudgetUsed += allcurrfidelity["{}_{}".format(of,ofi)] - atfid
            for numinner in range(num+1,len(origfile)):
                ofinner = origfile[numinner]
                if ofinner in of_done:
                    continue
                if ofinner == of:
                    ofiinner = origfileindex[numinner]
                    atfidinner = atfidelity[numinner]
                    ds["at fidelity"][ofiinner] = allcurrfidelity["{}_{}".format(ofinner,ofiinner)]
                    simulationBudgetUsed += allcurrfidelity["{}_{}".format(ofinner,ofiinner)] - atfidinner

            with open(of,'w') as f:
                json.dump(ds,f,indent=4)
            of_done.append(of)

    simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)

    if debug:
        print("mc_miniapp done.")
        sys.stdout.flush()
    return simulationBudgetUsed

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Simulation (with noise added)',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-o", dest="OPTION", type=str, default=None,
                        help="Option (initial,single, or multi)")
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")
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
                ato.writePythiaFiles(args.PROCESSCARD, param_names, [tr_center],
                                 outdir)

            simulationBudgetUsed = problem_main_program(
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
            simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
            ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                               value=simulationBudgetUsed)
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

        if args.OPTION == "multi":
            prevparamfile = "logs/prevparams_Np" + "_k{}.json".format(k)
            paramfile = newparams_Np_k
            outfile = MCout_Np_k
            outdir = outdir_Np_k
            if not gradCond:
                simulationBudgetUsed = problem_main_program(
                    paramfile,
                    prevparamfile,
                    args.WEIGHTS,
                    memorymap,
                    args.BEBOP,
                    outfile,
                    outdir
                )
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.writeMemoryMap(memoryMap=memorymap)
        else:
            prevparamfile = None
            paramfile = newparams_1_kp1
            outfile = MCout_1_kp1
            outdir = outdir_1_kp1
            simulationBudgetUsed = 0
            if not gradCond:
                if rank >= 0:
                    simulationBudgetUsed = problem_main_program(
                        paramfile,
                        prevparamfile,
                        args.WEIGHTS,
                        memorymap,
                        args.BEBOP,
                        outfile,
                        outdir
                    )
                simulationBudgetUsed = comm.bcast(simulationBudgetUsed, root=0)
                ato.putInMemoryMap(memoryMap=memorymap, key="simulationbudgetused",
                                   value=simulationBudgetUsed)
                ato.writeMemoryMap(memoryMap=memorymap)






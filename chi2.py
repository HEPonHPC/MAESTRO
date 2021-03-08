import pyhenson as h
import json
import numpy as np
import argparse
import sys
import apprentice

#orc@26-02:for now, putting this here, but then will be called via apprentice
def writePythiaFiles(proccardfile, pnames, points, outdir, fnamep="params.dat", fnameg="generator.cmd"):
     def readProcessCard(fname):
         with open(fname) as f:
             L = [l.strip() for l in f]
         return L
     from os.path import join, exists
     for num, p in enumerate(points):
         npad = "{}".format(num).zfill(1+int(np.ceil(np.log10(len(points)))))
         outd = join(outdir, npad)
         if not exists(outd):
             import os
             os.makedirs(outd)

         outfparams = join(outd, fnamep)
         with open(outfparams, "w") as pf:
             for k, v in zip(pnames, p):
                 pf.write("{name} {val:e}\n".format(name=k, val=v))

         outfgenerator = join(outd, fnameg)
         pc = readProcessCard(proccardfile)
         with open(outfgenerator, "w") as pg:
             for l in pc:
                 pg.write(l+"\n")
             for k, v in zip(pnames, p):
                 pg.write("{name} = {val:e}\n".format(name=k, val=v))

def mkCov(yerrs):
    import numpy as np
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

def run_chi2_optimization(algoparams,processcard,valfile,errfile,expdatafile,wtfile,chi2resultoutfile,pstarfile,pythiadir):
    #print("Starting chi2 optimization --")
    #sys.stdout.flush()
    with open(algoparams, 'r') as f:
        algoparamds = json.load(f)
    paramnames = algoparamds["param_names"]

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)

    debug = h.get("debug")

    res = IO.minimize(5,10)
    SCLR = IO._AS._RA[0]
    outputdata = {
        "x": res['x'].tolist(),
        "fun" : res['fun'],
        "scaler":SCLR.asDict
    }
    with open(chi2resultoutfile,'w') as f:
        json.dump(outputdata,f,indent=4)

    outds = {
        "parameters": [outputdata['x']]
    }
    if debug==1: print("\\SP amin \t= {}".format(["%.3f"%(c) for c in res['x']]))
    with open(pstarfile,'w') as f:
        json.dump(outds,f,indent=4)

    #TODO: shift to apprentice version when you have the latest version
    #apprentice.tools.writePythiaFiles(processcard,paramnames, [outputdata['x']], pythiadir)
    writePythiaFiles(processcard,paramnames, [outputdata['x']], pythiadir)

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve TR Subproblem',
                                     formatter_class=SaneFormatter)
#    parser.add_argument("--valappfile", dest="VALAPPFILE", type=str, default=None,
#                        help="Value approximation file name (JSON)")
#    parser.add_argument("--errappfile", dest="ERRAPPFILE", type=str, default=None,
#                        help="Error approximation file name (JSON)")
    parser.add_argument("-e", dest="EXPDATA", type=str, default=None,
                        help="Experimental data file (JSON)")
    parser.add_argument("-w", dest="WEIGHTS", type=str, default=None,
                        help="Weights file (TXT)")
#    parser.add_argument("--chi2resultfile", dest="CHI2RESULTFILE", type=str, default=None,
#                        help="Result ouput file (JSON)")
#    parser.add_argument("--pstarfile", dest="PSTARFILE", type=str, default=None,
#                        help="p^* parameter outfile (JSON)")
    parser.add_argument("-c", dest="PROCESSCARD", type=str, default=None,
                        help="Process Card location")
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON)")

    args = parser.parse_args()
    gradCond = h.get("signal")

    k = h.get("iter")
    newparams_1_kp1 = "logs/newparams_1" + "_k{}.json".format(k + 1)
    valapproxfile_k = "logs/valapprox" + "_k{}.json".format(k) 
    errapproxfile_k = "logs/errapprox" + "_k{}.json".format(k)
    resultoutfile_k = "logs/chi2result" + "_k{}.json".format(k)
    pythiadir_1_kp1 = "logs/pythia_1" + "_k{}".format(k+1)

    if gradCond ==0:
        run_chi2_optimization(
            args.ALGOPARAMS,
            args.PROCESSCARD,
            #args.VALAPPFILE,
            #args.ERRAPPFILE,
            valapproxfile_k,
            errapproxfile_k,
            args.EXPDATA,
            args.WEIGHTS,
            #args.CHI2RESULTFILE,
            resultoutfile_k,
            newparams_1_kp1,
            pythiadir_1_kp1
            #args.PSTARFILE
        )

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

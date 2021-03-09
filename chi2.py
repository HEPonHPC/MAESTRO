import json
import numpy as np
import argparse
import sys
import apprentice.tools as ato
import apprentice

def mkCov(yerrs):
    import numpy as np
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

def run_chi2_optimization(processcard,memorymap,valfile,errfile,
                          expdatafile,wtfile,chi2resultoutfile,pstarfile,pythiadir):
    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")
    if debug: print("Starting chi2 optimization --")
    sys.stdout.flush()

    param_names = ato.getFromMemoryMap(memoryMap=memorymap, key="param_names")

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valfile,
                                            errfile)

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

    ato.writePythiaFiles(processcard,param_names, [outputdata['x']], pythiadir)

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
    # parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
    #                     help="Algorithm Parameters (JSON)")

    args = parser.parse_args()

    (memorymap, pyhenson) = ato.readMemoryMap()
    k = ato.getFromMemoryMap(memoryMap=memorymap, key="iterationNo")
    gradCond = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_gradientCondition")

    newparams_1_kp1 = "logs/newparams_1" + "_k{}.json".format(k + 1)
    valapproxfile_k = "logs/valapprox" + "_k{}.json".format(k) 
    errapproxfile_k = "logs/errapprox" + "_k{}.json".format(k)
    resultoutfile_k = "logs/chi2result" + "_k{}.json".format(k)
    pythiadir_1_kp1 = "logs/pythia_1" + "_k{}".format(k + 1)

    if not gradCond:
        run_chi2_optimization(
            args.PROCESSCARD,
            memorymap,
            valapproxfile_k,
            errapproxfile_k,
            args.EXPDATA,
            args.WEIGHTS,
            resultoutfile_k,
            newparams_1_kp1,
            pythiadir_1_kp1
        )

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

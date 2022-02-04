import json
import numpy as np
import apprentice.tools as ato
import apprentice
import os,sys
from subprocess import Popen, PIPE
def getSamples():
    dim = 3
    n_p = 100
    minarr = [0,0.2,0.]
    maxarr = [2.,2.,1.]

    return ato.getLHSsamples(dim=dim, npoints=n_p, criterion="maximin",
                      minarr=minarr,maxarr=maxarr)

if __name__ == "__main__":

    MPATH = "/Users/mkrishnamoorthy/Research/Code/3Dminiapp/pythia8rivetminiapp/miniapp"
    weights = "../Parameters/Miniapp/weights"
    expdata = "../Parameters/Miniapp/data.json"

    outdir = "WD"
    yodaoutdir = os.path.join(outdir, "YODA")
    os.makedirs(yodaoutdir,exist_ok=True)
    fidelity = 10000
    X = getSamples()
    param_names = ["P1","P2","P3"]
    ato.writePythiaFiles("../process.dat", param_names, X, yodaoutdir, fnamep="params.dat",
                         fnameg="generator.cmd")
    import glob, os, re
    pfname = "params.dat"
    INDIRSLIST = [glob.glob(os.path.join(yodaoutdir, "*"))]

    for num, d in enumerate(INDIRSLIST[0]):
        files = glob.glob(os.path.join(d, "*"))
        param = None
        re_pfname = re.compile(pfname) if pfname else None
        for f in files:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                param = apprentice.io.read_paramsfile(f)

        newloc = os.path.join(d, "ne{}_p{}.yoda".format(fidelity, str(num + 1)))
        pp = [param[pn] for pn in param_names]
        print(pp)
        p = Popen(
            [MPATH, str(pp[0]), str(pp[1]), str(pp[2]),
             str(fidelity), str(874673), "0", "1", newloc],
            stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        if p.returncode != 0:
            raise Exception("Running miniapp failed with return code {}".format(p.returncode))

    DATAprev, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
        [yodaoutdir], "params.dat", weights)

    valapp = {}
    errapp = {}
    for num, (X, Y, E) in enumerate(DATAprev):
        thisBinId = binids[num]
        val = apprentice.RationalApproximation(X, Y, order=(2, 0), pnames=pnames)
        err = apprentice.RationalApproximation(X, E, order=(1, 0), pnames=pnames)
        valapp[thisBinId] = val.asDict
        errapp[thisBinId] = err.asDict
    from collections import OrderedDict

    valoutfile = os.path.join(outdir,"val.json")
    erroutfile = os.path.join(outdir,"err.json")
    with open(valoutfile, "w") as f:
        json.dump(valapp, f, indent=4)
    with open(erroutfile, "w") as f:
        json.dump(errapp, f, indent=4)

    IO = apprentice.appset.TuningObjective2(weights,
                                            expdata,
                                            valoutfile,
                                            erroutfile,
                                            debug=False)

    res = IO.minimizeMPI(nstart=5,nrestart=100,saddlePointCheck=False)
    print(res)










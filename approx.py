import h5py
import apprentice
import json
import sys
import argparse
import numpy as np
import apprentice.tools as ato

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
def run_approx(memorymap,interpolationdatafile,valoutfile, erroutfile,expdatafile,wtfile):
    debug = ato.getFromMemoryMap(memoryMap=memorymap, key="debug")

    if debug: print("Starting approximation --")
    # print("CHANGE ME TO THE PARALLEL VERSION")
    assert (erroutfile != interpolationdatafile)
    assert (valoutfile != interpolationdatafile)

    DATA = apprentice.io.readH5(interpolationdatafile)
    # print(DATA)
    pnames = apprentice.io.readPnamesH5(interpolationdatafile, xfield="params")
    idx = [i for i in range(len(DATA))]
    valapp = []
    errapp = []

    # S = apprentice.Scaler(DATA[0][0])  # Let's assume that all X are the same for simplicity
    # print("Halfway reporting: before generating the output file --")
    for num, (_X, _Y, _E) in enumerate(DATA):
        # print(_X)
        # print(_Y)
        try:
            # print("\n\n\n\n")
            # print(_X,_Y,_E)
            valapp.append(apprentice.RationalApproximation(_X, _Y, order=(2, 0), pnames=pnames))
            errapp.append(apprentice.RationalApproximation(_X, _E, order=(1, 0), pnames=pnames))
        except AssertionError as error:
            print(error)
    # What do these do? Are the next 4 lines required
    # S.save("{}.scaler".format(valoutfile))
    # S.save("{}.scaler".format(erroutfile))
    # S.save(valoutfile)
    # S.save(erroutfile)

    # This reads the unique identifiers of the bins
    with h5py.File(interpolationdatafile, "r") as f:
        binids = f.get("index")[idx]

    JD = {x.decode(): y.asDict for x, y in zip(binids, valapp)}
    with open(valoutfile, "w") as f:
        json.dump(JD, f)

    JD = {x.decode(): y.asDict for x, y in zip(binids, errapp)}
    with open(erroutfile, "w") as f:
        json.dump(JD, f)

    # print("Done --- approximation of {} objects written to {} and {}".format(
    #         len(idx), valoutfile, erroutfile))

    tr_radius = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_radius")
    tr_center = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_center")
    tr_sigma = ato.getFromMemoryMap(memoryMap=memorymap, key="tr_sigma")


    #print("BYE from approx")
    #sys.stdout.flush()

    IO = apprentice.appset.TuningObjective2(wtfile,
                                            expdatafile,
                                            valoutfile,
                                            erroutfile)
    IO._AS.setRecurrence(tr_radius)
    IO._EAS.setRecurrence(tr_radius)
    grad = IO.gradient(tr_center)
    if np.linalg.norm(grad) <= tr_sigma * tr_radius:
        ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                           value=True)
    else: ato.putInMemoryMap(memoryMap=memorymap, key="tr_gradientCondition",
                       value=False)

    if debug: print("||grad|| \t= %.3f <=> %.3f"%(np.linalg.norm(grad),tr_sigma * tr_radius))
    sys.stdout.flush()
    ato.writeMemoryMap(memorymap)

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

    MCout_Np_k = "logs/MCout_Np" + "_k{}.h5".format(k)
    valapproxfile_k = "logs/valapprox" + "_k{}.json".format(k)
    errapproxfile_k = "logs/errapprox" + "_k{}.json".format(k)
    run_approx(
        memorymap,
        MCout_Np_k,
        valapproxfile_k,
        errapproxfile_k,
        args.EXPDATA,
        args.WEIGHTS
    )

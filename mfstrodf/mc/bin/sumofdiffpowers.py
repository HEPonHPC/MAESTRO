import argparse
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run sum of diff powers',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-d", dest="MCDIR", type=str, default="log/MC_RUN",
                        help="MC directory")

    args = parser.parse_args()

    from mfstrodf.mc import SumOfDiffPowers
    mctask = SumOfDiffPowers(args.MCDIR)
    mctask.run_mc()
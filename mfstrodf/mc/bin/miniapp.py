import argparse
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run sum of diff powers',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-d", dest="MCDIR", type=str, default="log/MC_RUN",
                        help="MC directory")
    parser.add_argument("-c", dest="CONFIG", type=str, default=None,
                        help="Config file location")

    args = parser.parse_args()
    import json
    with open(args.CONFIG,'r') as f:
        ds = json.load(f)
    mc_parameters = ds['mc']['parameters']

    from mfstrodf.mc import MiniApp
    mctask = MiniApp(args.MCDIR,mc_parameters)
    mctask.run_mc()
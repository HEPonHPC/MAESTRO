from mfstrodf.mc import MCTask,DiskUtil
import numpy as np
import pprint,os

class SimpleApp(MCTask):
    def run_mc(self):
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        (term_names,term_class) = self.get_functions()
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class
            if run_fidelity>0:
                Y = [np.random.normal(clss.mapping(param),
                                      1 / np.sqrt(run_fidelity), 1)[0]
                                        for clss in term_class]
                DY = [1 / np.sqrt(run_fidelity) for clss in term_class]

                outfile = os.path.join(d,"out_curr.csv")
                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class

    def get_functions(self):
        term_names = []
        term_class = []
        count = {}
        import mfstrodf.mc
        for t in self.mc_parmeters['functions']:
            count[t] = 1 if t not in term_names else count[t] + 1
            sct = "" if count[t] == 1 else str(count[t])
            term_names.append("{}{}".format(t,sct))
            try:
                mc_class = getattr(mfstrodf.mc,t)
                term_class.append(mc_class)
            except:
                raise Exception("MC term class \""+t+"\" not found in mfstrodf.mc")
        return term_names,term_class

    def check_df_structure_sanity(self,df):
        rownames = list(df.columns.values)
        columnnames = list(df.index)
        if len(rownames)>1 and ('.P' not in rownames[0] and '.V' not in rownames[1]) and \
                len(columnnames)>1 and ('.P' not in columnnames[0] and '.V' not in columnnames[1]):
            raise Exception('The MC data frame does not have a parameter index that ends in \".P\" '
                            'and value index that ends in \".V\"')
        if len(rownames)>1 and ('.P' in rownames[0] and '.V' in rownames[1]):
            df = df.transpose()
        return df

    def convert_mc_output_to_df(self, all_param_directory):
        df = self.convert_csv_data_to_df(all_param_directory=all_param_directory,
                                         mc_out_file_name="out.csv")
        df = self.check_df_structure_sanity(df)
        additional_data = None
        return (df,additional_data)

    def merge_statistics_and_get_max_sigma(self):
        import pandas as pd
        import math
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        (term_names,term_class) = self.get_functions()
        max_sigma = 0
        for dno,d in enumerate(dirlist):
            at_fidelity = self.get_fidelity_from_directory(d,fnamef="at_fidelity.dat")
            run_fidelity = self.get_fidelity_from_directory(d)
            prev_mc_out_path = os.path.join(d,"out.csv")
            curr_mc_out_path = os.path.join(d,"out_curr.csv")
            if not os.path.exists(prev_mc_out_path):
                DiskUtil.copyanything(curr_mc_out_path,prev_mc_out_path)
            if not os.path.exists(curr_mc_out_path):
                DiskUtil.copyanything(prev_mc_out_path,curr_mc_out_path)
            curr_df = pd.read_csv(curr_mc_out_path)
            rownames = list(curr_df.columns.values)
            columnnames = list(curr_df.index)
            prev_df = pd.read_csv(prev_mc_out_path)
            _Y = []
            _E = []

            for cno in range(len(columnnames)):
                curr_val = curr_df[rownames[1]][columnnames[cno]]
                prev_val = prev_df[rownames[1]][columnnames[cno]]
                if not math.isnan(curr_val) and not math.isinf(curr_val) \
                    and not math.isnan(prev_val) and not math.isinf(prev_val):
                    _Y.append(np.average([curr_val,prev_val]))
                else:
                    _Y.append(np.nan)
                _E.append(1 / np.sqrt((at_fidelity+run_fidelity)))
            self.save_mc_out_as_csv(header="name,MC,DMC",
                                    term_names=term_names,data=[_Y,_E],out_path=prev_mc_out_path
                                    ) # from super class
            DiskUtil.remove_file(curr_mc_out_path)
            max_sigma = max(max_sigma,np.max(_E))
        return max_sigma

class DeterministicApp(SimpleApp):
    def run_mc(self):
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        (term_names,term_class) = self.get_functions() # from super class
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class
            if run_fidelity>0:
                Y = [clss.mapping(param) for clss in term_class]
                DY = [1. for clss in term_class]

                outfile = os.path.join(d,"out_curr.csv")
                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class

class SumOfDiffPowers():
    @staticmethod
    def mapping(x):
        # https://www.sfu.ca/~ssurjano/sumpow.html
        s = 0
        for i in range(len(x)):
            n = (abs(x[i])) ** (i + 2)
            s = s + n
        return s

class RotatedHyperEllipsoid():
    @staticmethod
    def mapping(x):
        # https://www.sfu.ca/~ssurjano/rothyp.html
        outer = 0.
        for i in range(len(x)):
            inner = 0.
            for j in range(i):
                inner = inner + x[j]**2
            outer = outer + inner
        return outer

class Sphere():
    @staticmethod
    def mapping(x):
        # https://www.sfu.ca/~ssurjano/spheref.html
        s = 0
        for i in range(len(x)):
            s = s + x[i]**2
        return s


class SumSquares():
    @staticmethod
    def mapping(x):
        # https://www.sfu.ca/~ssurjano/sumsqu.html
        s = 0
        for i in range(len(x)):
            s = s + (i+1)*x[i]**2
        return s










